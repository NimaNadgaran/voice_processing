"""Quality metrics.

Two families live here:

* **Reference-free** (always computed) -- we normally do *not* have the clean
  ground truth for a user upload, so we estimate quality from the signal itself:
  VAD-based SNR, noise-floor level, speech ratio, cross-talk between separated
  tracks, speaker-distinctness.  These are honest *proxies*, not oracle scores;
  the frontend labels them "estimated".
* **Reference-based** (only when a clean reference exists, e.g. in the training
  / evaluation scripts) -- SI-SDR, SI-SNR improvement, PESQ, STOI.

See ``src/denoising/README.md`` and ``src/separation/README.md`` for how each
number should be read.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .dsp import EPS, energy_vad, stft
from .types import AudioBuffer
from .utils import module_available


# --------------------------------------------------------------------------- #
#  single-signal descriptors
# --------------------------------------------------------------------------- #
def signal_stats(audio: AudioBuffer) -> Dict[str, Any]:
    x = audio.samples
    if x.size == 0:
        return {"duration": 0.0, "rms_db": -120.0, "peak_db": -120.0, "crest_db": 0.0}
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    return {
        "duration": round(audio.duration, 3),
        "sample_rate": audio.sr,
        "rms_db": round(float(20 * math.log10(max(rms, EPS))), 2),
        "peak_db": round(float(20 * math.log10(max(peak, EPS))), 2),
        "crest_db": round(float(20 * math.log10(max(peak, EPS) / max(rms, EPS))), 2),
        "clipping_ratio": round(float(np.mean(np.abs(x) > 0.999)), 5),
        "dc_offset": round(float(np.mean(x)), 6),
    }


def estimated_snr(audio: AudioBuffer) -> Dict[str, float]:
    """Reference-free SNR estimate.

    Primary method: VAD-based -- compare the power of speech frames against the
    power of the silence frames, which *are* the noise.

    Fallback: when the clip contains no silence at all (someone talking without
    pause, or speech over continuous music) there are no noise-only frames to
    measure, so we switch to **minimum statistics**: the 10th percentile over
    time of each frequency bin is treated as that bin's noise floor. Without
    this, a gap-free signal reports a meaningless SNR near the -20 dB clamp.
    """
    x = audio.samples
    if x.size < audio.sr // 10:
        return {"snr_db": 0.0, "noise_floor_db": -120.0, "speech_ratio": 0.0, "speech_seconds": 0.0}

    mask, hop_s = energy_vad(x, audio.sr)
    frame_len = max(1, int(audio.sr * 0.03))
    hop = max(1, int(audio.sr * 0.01))
    n = len(mask)
    powers = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop: i * hop + frame_len]
        powers[i] = float(np.mean(seg**2)) if seg.size else EPS

    # Both classes must hold a meaningful share of the clip. A handful of
    # "silence" frames in an otherwise gap-free recording are usually just the
    # edges of speech, and averaging them gives a noise floor equal to the
    # speech level -- which reports a nonsense SNR at the clamp.
    speech_share, silence_share = float(mask.mean()), float((~mask).mean())
    if speech_share >= 0.10 and silence_share >= 0.10:
        speech_p = float(np.mean(powers[mask]))
        noise_p = max(float(np.mean(powers[~mask])), EPS)
        method = "vad"
    else:
        # No usable silence. Estimate what fraction of the power is noise, two
        # independent ways, and believe the more optimistic one:
        #   * over TIME  -- the 10th percentile of each bin (minimum statistics);
        #     blind to stationary interference, since a constant tone looks like
        #     a constant floor.
        #   * over FREQUENCY -- the 25th percentile across bins in each frame;
        #     noise is broadband, speech and tones are peaky, so the quiet bins
        #     between harmonics are mostly noise.
        # Neither alone is reliable; the minimum of the two avoids the failure
        # mode where a clean stationary signal is declared to be all noise.
        # Bias correction: the periodogram of Gaussian noise is exponentially
        # distributed, so its p-th quantile sits at -ln(1-p) of the true mean,
        # not at the mean. Without dividing that out, both estimators
        # under-report the noise by ~5-10 dB.
        Q10, Q25 = -math.log(1 - 0.10), -math.log(1 - 0.25)  # 0.105, 0.288

        spec_power = np.abs(stft(x, n_fft=1024, hop=512)) ** 2
        mean_power = float(spec_power.mean()) + EPS

        temporal = float(np.percentile(spec_power, 10.0, axis=1).mean()) / (mean_power * Q10)
        spectral = float(np.percentile(spec_power, 25.0, axis=0).mean()) / (mean_power * Q25)

        noise_fraction = float(np.clip(min(temporal, spectral), 1e-8, 1.0))
        total_power = max(float(np.mean(x**2)), EPS)
        speech_p = total_power
        noise_p = max(total_power * noise_fraction, EPS)
        method = "minimum-statistics"

    snr = 10.0 * math.log10(max(speech_p - noise_p, EPS) / noise_p)
    return {
        "snr_db": round(float(np.clip(snr, -20.0, 80.0)), 2),
        "noise_floor_db": round(float(10 * math.log10(max(noise_p, EPS))), 2),
        "speech_ratio": round(float(mask.mean()) if n else 0.0, 4),
        "speech_seconds": round(float(mask.sum() * hop_s), 2),
        "snr_method": method,
    }


def spectral_profile(audio: AudioBuffer, n_bands: int = 8) -> List[float]:
    """Average energy per octave-ish band in dB -- drives the UI spectrum bars."""
    x = audio.samples
    if x.size == 0:
        return [-120.0] * n_bands
    spec = np.abs(stft(x, n_fft=1024, hop=512))
    mean_mag = spec.mean(axis=1)
    edges = np.geomspace(1, spec.shape[0] - 1, n_bands + 1).astype(int)
    out = []
    for i in range(n_bands):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out.append(round(float(20 * math.log10(max(float(mean_mag[lo:hi].mean()), EPS))), 2))
    return out


# --------------------------------------------------------------------------- #
#  denoising: before / after
# --------------------------------------------------------------------------- #
def denoise_metrics(before: AudioBuffer, after: AudioBuffer) -> Dict[str, Any]:
    a, b = estimated_snr(before), estimated_snr(after)
    stats_b = signal_stats(after)

    residual = _aligned_residual(before, after)
    removed_db = float(20 * math.log10(max(float(np.sqrt(np.mean(residual**2))), EPS))) if residual.size else -120.0

    snr_gain = b["snr_db"] - a["snr_db"]
    noise_drop = a["noise_floor_db"] - b["noise_floor_db"]

    # artefact guard: did the speech *structure* survive the processing?
    speech_kept, band_ratio = _speech_preservation(before, after)

    score = _quality_score(snr_gain, noise_drop, speech_kept)
    return {
        "snr_before_db": a["snr_db"],
        "snr_after_db": b["snr_db"],
        "snr_improvement_db": round(snr_gain, 2),
        "noise_floor_before_db": a["noise_floor_db"],
        "noise_floor_after_db": b["noise_floor_db"],
        "noise_reduction_db": round(noise_drop, 2),
        "removed_energy_db": round(removed_db, 2),
        "speech_preserved": round(speech_kept, 4),
        "speech_band_energy_ratio": round(band_ratio, 4),
        "speech_ratio_after": b["speech_ratio"],
        "quality_score": round(score, 1),
        "output_rms_db": stats_b["rms_db"],
        "output_peak_db": stats_b["peak_db"],
        "clipping_ratio": stats_b["clipping_ratio"],
    }


def _aligned_residual(before: AudioBuffer, after: AudioBuffer) -> np.ndarray:
    x, y = before.samples, after.samples
    if before.sr != after.sr:
        from .audio_io import resample

        y = resample(after, before.sr).samples
    n = min(len(x), len(y))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return (x[:n] - y[:n]).astype(np.float32)


def _speech_preservation(
    before: AudioBuffer, after: AudioBuffer, lo_hz: float = 300.0, hi_hz: float = 3400.0
) -> Tuple[float, float]:
    """Did the *speech structure* survive?  Returns (correlation, energy ratio).

    A denoiser is supposed to remove energy, so a plain before/after energy ratio
    punishes good denoisers (it drops as soon as the hiss goes).  What actually
    signals damage is the speech **envelope** changing shape: if the frame-by-frame
    energy contour of the telephone band still tracks the original during speech,
    the words are intact no matter how much noise was stripped.

    * ``correlation`` ~1.0  -> the speech envelope is preserved (good)
    * ``correlation`` <0.7  -> the processor is chewing into the speech
    * ``energy_ratio``      -> reported separately, purely informational
    """
    from .audio_io import resample

    y = after if after.sr == before.sr else resample(after, before.sr)
    n = min(before.n_samples, y.n_samples)
    if n < before.sr // 5:
        return 1.0, 1.0

    n_fft = 1024
    sx = np.abs(stft(before.samples[:n], n_fft=n_fft, hop=n_fft // 2))
    sy = np.abs(stft(y.samples[:n], n_fft=n_fft, hop=n_fft // 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / before.sr)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    frames = min(sx.shape[1], sy.shape[1])
    if frames < 4:
        return 1.0, 1.0

    ex = (sx[band, :frames] ** 2).sum(axis=0)
    ey = (sy[band, :frames] ** 2).sum(axis=0)
    energy_ratio = float(np.clip(ey.sum() / (ex.sum() + EPS), 0.0, 4.0))

    # only judge frames that actually carried speech in the original
    keep = ex > np.percentile(ex, 55.0)
    if keep.sum() < 4:
        keep = np.ones_like(ex, dtype=bool)
    a = np.log(np.maximum(ex[keep], EPS))
    b = np.log(np.maximum(ey[keep], EPS))
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2))) + EPS
    corr = float(np.clip(np.sum(a * b) / denom, 0.0, 1.0))
    return corr, energy_ratio


def _quality_score(snr_gain: float, noise_drop: float, speech_kept: float) -> float:
    """Blend the proxies into a friendly 0-100 "how well did it go" number."""
    gain_part = np.clip(snr_gain / 15.0, 0.0, 1.0) * 45.0
    drop_part = np.clip(noise_drop / 20.0, 0.0, 1.0) * 30.0
    keep_part = np.clip((speech_kept - 0.5) / 0.45, 0.0, 1.0) * 25.0
    return float(np.clip(gain_part + drop_part + keep_part, 0.0, 100.0))


# --------------------------------------------------------------------------- #
#  separation quality (reference free)
# --------------------------------------------------------------------------- #
def separation_metrics(mixture: AudioBuffer, tracks: Sequence[AudioBuffer]) -> Dict[str, Any]:
    if not tracks:
        return {"n_sources": 0, "separation_score": 0.0}

    n = min([mixture.n_samples] + [t.n_samples for t in tracks])
    stack = np.stack([t.samples[:n] for t in tracks], axis=0) if n else np.zeros((len(tracks), 1), np.float32)

    # pairwise correlation -> how much the tracks still look like each other
    corr = _pairwise_corr(stack)
    off = corr[~np.eye(len(tracks), dtype=bool)] if len(tracks) > 1 else np.zeros(1)
    mean_leak = float(np.mean(np.abs(off))) if off.size else 0.0
    max_leak = float(np.max(np.abs(off))) if off.size else 0.0

    # energy conservation: sum of sources vs mixture
    mix = mixture.samples[:n]
    recon = stack.sum(axis=0)
    mix_e = float(np.sum(mix**2)) + EPS
    err = float(np.sum((mix - recon) ** 2))
    conservation = float(np.clip(1.0 - err / mix_e, -1.0, 1.0))

    per_track = []
    for i, t in enumerate(tracks):
        s = estimated_snr(t)
        per_track.append(
            {
                "index": i,
                "rms_db": signal_stats(t)["rms_db"],
                "speech_ratio": s["speech_ratio"],
                "speech_seconds": s["speech_seconds"],
                "snr_db": s["snr_db"],
            }
        )

    score = float(np.clip((1.0 - mean_leak) * 70.0 + max(conservation, 0.0) * 30.0, 0.0, 100.0))
    return {
        "n_sources": len(tracks),
        "mean_cross_correlation": round(mean_leak, 4),
        "max_cross_correlation": round(max_leak, 4),
        "energy_conservation": round(conservation, 4),
        "separation_score": round(score, 1),
        "per_track": per_track,
    }


def _pairwise_corr(stack: np.ndarray) -> np.ndarray:
    x = stack - stack.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(x**2, axis=1, keepdims=True)) + EPS
    x = x / norm
    return np.clip(x @ x.T, -1.0, 1.0)


# --------------------------------------------------------------------------- #
#  reference-based (training / evaluation only)
# --------------------------------------------------------------------------- #
def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al., 2019)."""
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + EPS))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))


def permutation_si_sdr(estimates: Sequence[np.ndarray], references: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Best-permutation SI-SDR (what PIT training optimises)."""
    import itertools

    best = None
    for perm in itertools.permutations(range(len(references))):
        vals = [si_sdr(estimates[i], references[p]) for i, p in enumerate(perm)]
        mean_val = float(np.mean(vals))
        if best is None or mean_val > best[0]:
            best = (mean_val, perm, vals)
    return {"si_sdr": round(best[0], 3), "permutation": list(best[1]), "per_source": [round(v, 3) for v in best[2]]}


def pesq_score(ref: AudioBuffer, deg: AudioBuffer) -> Optional[float]:
    if not module_available("pesq"):
        return None
    try:
        from pesq import pesq as _pesq

        from .audio_io import resample

        sr = 16000
        r = resample(ref, sr).samples
        d = resample(deg, sr).samples
        n = min(len(r), len(d))
        return round(float(_pesq(sr, r[:n], d[:n], "wb")), 3)
    except Exception:
        return None


def stoi_score(ref: AudioBuffer, deg: AudioBuffer) -> Optional[float]:
    if not module_available("pystoi"):
        return None
    try:
        from pystoi import stoi as _stoi

        from .audio_io import resample

        sr = 16000
        r = resample(ref, sr).samples
        d = resample(deg, sr).samples
        n = min(len(r), len(d))
        return round(float(_stoi(r[:n], d[:n], sr, extended=False)), 4)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  fun facts for the UI
# --------------------------------------------------------------------------- #
def fun_facts(audio: AudioBuffer, tracks: Sequence[AudioBuffer]) -> Dict[str, Any]:
    """Light-weight "nice to know" numbers rendered as cards in the frontend."""
    snr = estimated_snr(audio)
    total = audio.duration or 1.0
    talk = []
    for t in tracks:
        s = estimated_snr(t)
        talk.append(s["speech_seconds"])
    total_talk = sum(talk) or 1.0

    # crude interruption / overlap proxy: sum of per-speaker speech vs clip length
    overlap_ratio = float(np.clip(total_talk / total - 1.0, 0.0, 5.0)) if tracks else 0.0

    return {
        "clip_seconds": round(total, 2),
        "silence_percent": round(100.0 * (1.0 - snr["speech_ratio"]), 1),
        "talk_time": [round(v, 2) for v in talk],
        "talk_share_percent": [round(100.0 * v / total_talk, 1) for v in talk],
        "most_talkative": int(np.argmax(talk)) if talk else None,
        "estimated_overlap_percent": round(100.0 * overlap_ratio, 1),
        "words_estimate": int(total_talk * 2.5),  # ~150 wpm
        "megabytes_of_pcm": round(audio.n_samples * 4 / (1024 * 1024), 2),
    }
