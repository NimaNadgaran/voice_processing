"""Pure-numpy DSP layer: STFT/ISTFT, VAD, noise PSD estimation, MFCC features.

Everything degrades to numpy so the "always available" denoiser and separator
work on a bare install.  scipy / librosa are used opportunistically when present.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

EPS = 1e-10


# --------------------------------------------------------------------------- #
#  windows / framing
# --------------------------------------------------------------------------- #
def make_window(n: int, kind: str = "hann") -> np.ndarray:
    if kind == "hann":
        return np.hanning(n + 1)[:-1].astype(np.float32)
    if kind == "sqrt_hann":
        return np.sqrt(np.hanning(n + 1)[:-1]).astype(np.float32)
    if kind == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """(n_frames, frame_len) view-ish array, zero padded at the tail."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


# --------------------------------------------------------------------------- #
#  STFT / ISTFT  (centred, hann, weighted overlap-add)
# --------------------------------------------------------------------------- #
def stft(
    x: np.ndarray,
    n_fft: int = 1024,
    hop: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Return a complex spectrogram of shape (n_fft // 2 + 1, n_frames)."""
    hop = hop or n_fft // 4
    x = np.asarray(x, dtype=np.float32)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect" if len(x) > n_fft else "constant")
    win = make_window(n_fft, window)
    frames = frame_signal(x, n_fft, hop) * win[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T.astype(np.complex64)


def istft(
    spec: np.ndarray,
    hop: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    length: Optional[int] = None,
) -> np.ndarray:
    """Inverse of :func:`stft` using weighted overlap-add (NOLA safe)."""
    n_fft = 2 * (spec.shape[0] - 1)
    hop = hop or n_fft // 4
    win = make_window(n_fft, window)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1).astype(np.float32)

    n_frames = frames.shape[0]
    out_len = n_fft + hop * (n_frames - 1)
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    win_sq = win**2
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * win
        wsum[s:s + n_fft] += win_sq
    out /= np.maximum(wsum, 1e-8)

    if center:
        out = out[n_fft // 2:]
        if length is None:
            out = out[: max(0, out_len - n_fft)]
    if length is not None:
        out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
    return out.astype(np.float32)


def magphase(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.abs(spec).astype(np.float32)
    phase = np.exp(1j * np.angle(spec)).astype(np.complex64)
    return mag, phase


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


# --------------------------------------------------------------------------- #
#  noise estimation
# --------------------------------------------------------------------------- #
def estimate_noise_psd(mag: np.ndarray, percentile: float = 15.0, n_init_frames: int = 12) -> np.ndarray:
    """Per-bin noise magnitude estimate.

    Uses a low percentile over time (robust "minimum statistics" style estimate)
    blended with the first frames, which in real recordings are usually silence.
    """
    if mag.size == 0:
        return np.zeros((mag.shape[0],), dtype=np.float32)
    quiet = np.percentile(mag, percentile, axis=1).astype(np.float32)
    head = mag[:, : max(1, min(n_init_frames, mag.shape[1]))].mean(axis=1).astype(np.float32)
    return np.minimum(quiet * 1.0, np.maximum(quiet * 0.5, head)).astype(np.float32)


def spectral_flatness(mag: np.ndarray) -> np.ndarray:
    """Per-frame Wiener entropy in [0, 1]; ~1 = noise-like, ~0 = tonal/voiced."""
    m = np.maximum(mag, EPS)
    geo = np.exp(np.mean(np.log(m), axis=0))
    arith = np.mean(m, axis=0)
    return (geo / np.maximum(arith, EPS)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  voice activity detection (energy + flatness, hysteresis smoothed)
# --------------------------------------------------------------------------- #
def energy_vad(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_db: float = 18.0,
    min_speech_ms: float = 150.0,
    min_silence_ms: float = 120.0,
) -> Tuple[np.ndarray, float]:
    """Return (bool mask per hop-frame, hop seconds).

    ``threshold_db`` is measured *above the noise floor* of the clip, so it is
    level independent.
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames = frame_signal(x, frame_len, hop)
    if frames.size == 0:
        return np.zeros(0, dtype=bool), hop / float(sr)

    energy = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    floor = float(np.percentile(energy, 10))
    ceil_ = float(np.percentile(energy, 95))
    span = ceil_ - floor
    thr = floor + min(threshold_db, max(4.0, span * 0.45))
    mask = energy > thr

    # ------------------------------------------------------------------ #
    # A purely relative energy threshold assumes the clip contains silence to
    # measure the floor from. When someone talks with barely a pause -- which is
    # normal in an interview, and universal in concatenated corpus audio -- the
    # 10th percentile *is* speech, so the threshold sits inside the voice and
    # throws half of it away (measured: 99% speech scored as 49%, leaving too
    # few windows to cluster and wrecking the speaker count).
    #
    # Spectral flatness tells the two "no dynamic range" cases apart: speech is
    # harmonic and therefore peaky (low flatness), steady noise is broadband
    # (high flatness). So a quiet-but-tonal frame is kept as speech.
    # ------------------------------------------------------------------ #
    if span < 12.0 or mask.mean() < 0.75:
        n_fft = 512
        spec = np.abs(stft(x, n_fft=n_fft, hop=hop, center=True))
        # Judge tonality only inside the speech band. Measured over the whole
        # spectrum, a 50 Hz mains hum looks perfectly "tonal" and gets accepted
        # as speech; between 200 Hz and 4 kHz it has almost no energy, so it
        # correctly reads as empty.
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        band = (freqs >= 200.0) & (freqs <= 4000.0)
        flat = spectral_flatness(spec[band])[: len(energy)]
        if len(flat) < len(energy):
            flat = np.pad(flat, (0, len(energy) - len(flat)), mode="edge")

        # Tonality alone is not enough: a 50 Hz mains hum is extremely tonal and
        # its spectral leakage still looks peaky inside the band. Speech must
        # also actually *live* in the speech band, so require a real share of
        # the frame's energy to be there. Hum and rumble put ~0% there; speech
        # puts most of it there.
        power = spec**2
        band_ratio = power[band].sum(axis=0) / (power.sum(axis=0) + EPS)
        band_ratio = band_ratio[: len(energy)]
        if len(band_ratio) < len(energy):
            band_ratio = np.pad(band_ratio, (0, len(energy) - len(band_ratio)), mode="edge")

        # Adaptive tonality threshold: at low SNR every frame is noisier, so a
        # fixed cutoff would reject everything. Take the more tonal frames.
        cutoff = float(np.clip(np.percentile(flat, 55.0), 0.05, 0.45))
        voiced = (flat <= cutoff) & (band_ratio > 0.15)
        mask = mask | (voiced & (energy > floor - 1.0))

    min_speech = max(1, int(min_speech_ms / hop_ms))
    min_sil = max(1, int(min_silence_ms / hop_ms))
    mask = _fill_short(mask, False, min_sil)  # bridge tiny gaps
    mask = _fill_short(mask, True, min_speech)  # drop tiny blips
    return mask, hop / float(sr)


def _fill_short(mask: np.ndarray, value: bool, min_run: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == value:
            j = i
            while j < n and out[j] == value:
                j += 1
            if (j - i) < min_run and i > 0 and j < n:
                out[i:j] = not value
            i = j
        else:
            i += 1
    return out


def mask_to_segments(mask: np.ndarray, hop_s: float, min_dur: float = 0.15) -> List[List[float]]:
    segs: List[List[float]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            start, end = i * hop_s, j * hop_s
            if end - start >= min_dur:
                segs.append([float(start), float(end)])
            i = j
        else:
            i += 1
    return segs


def segments_to_mask(segments, n_samples: int, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    """Sample-level 0/1 mask with cosine fades (avoids clicks at boundaries)."""
    mask = np.zeros(n_samples, dtype=np.float32)
    fade = max(2, int(sr * fade_ms / 1000.0))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, fade, dtype=np.float32)))
    for start, end in segments:
        a = max(0, int(start * sr))
        b = min(n_samples, int(end * sr))
        if b <= a:
            continue
        mask[a:b] = 1.0
        f = min(fade, (b - a) // 2)
        if f > 1:
            mask[a:a + f] = np.minimum(mask[a:a + f], ramp[:f])
            mask[b - f:b] = np.minimum(mask[b - f:b], ramp[:f][::-1])
    return mask


# --------------------------------------------------------------------------- #
#  features for speaker clustering (MFCC + deltas, numpy implementation)
# --------------------------------------------------------------------------- #
def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 40.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    mels = np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left:
            centre = min(left + 1, n_fft // 2)
        if right == centre:
            right = min(centre + 1, n_fft // 2)
        for k in range(left, centre):
            fb[i, k] = (k - left) / max(1, centre - left)
        for k in range(centre, right):
            fb[i, k] = (right - k) / max(1, right - centre)
    return fb


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    mat = np.cos(math.pi / n_in * (n + 0.5) * k).astype(np.float32)
    mat[0] *= 1.0 / math.sqrt(2.0)
    return (mat * math.sqrt(2.0 / n_in)).astype(np.float32)


def mfcc(
    x: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop: Optional[int] = None,
    n_mels: int = 40,
) -> np.ndarray:
    """(n_mfcc, n_frames) MFCCs -- numpy only."""
    hop = hop or n_fft // 2
    spec = np.abs(stft(x, n_fft=n_fft, hop=hop, center=True))
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = np.maximum(fb @ (spec**2), EPS)
    log_mel = np.log(mel).astype(np.float32)
    return (dct_matrix(n_mfcc, n_mels) @ log_mel).astype(np.float32)


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(feat, ((0, 0), (width, width)), mode="edge")
    out = np.zeros_like(feat)
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    for i in range(1, width + 1):
        out += i * (padded[:, width + i: padded.shape[1] - width + i] - padded[:, width - i: padded.shape[1] - width - i])
    return (out / denom).astype(np.float32)


def cmvn(feat: np.ndarray) -> np.ndarray:
    """Cepstral mean & variance normalisation -- removes the channel/mic colour."""
    mu = feat.mean(axis=1, keepdims=True)
    sd = feat.std(axis=1, keepdims=True) + 1e-6
    return ((feat - mu) / sd).astype(np.float32)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same").astype(np.float32)
