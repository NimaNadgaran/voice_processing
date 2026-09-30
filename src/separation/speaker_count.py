"""How many people are talking?

Everything downstream needs this number, and the user usually does not know it,
so we estimate it and report a confidence.

Algorithm
---------
1. VAD -> keep only speech.
2. Slice the speech into short windows (default 1.5 s, 0.75 s hop).
3. Embed each window:
   * **ECAPA-TDNN** speaker embeddings (SpeechBrain) when installed -- these are
     trained to be speaker-discriminative and are dramatically better; else
   * a numpy fallback: CMVN-normalised MFCC mean/std + delta stats + pitch and
     spectral-slope descriptors.  Good enough to tell apart voices that differ
     in pitch/timbre, weaker on similar voices.
4. Cluster for every candidate K and score the partition with the silhouette
   coefficient; also compute the spectral **eigengap** of the affinity matrix.
5. Pick the K with the best combined score.  If the best silhouette is below
   ``single_speaker_threshold`` we call it one speaker.

The returned ``confidence`` is the normalised margin between the best and the
runner-up K -- so "4 speakers (0.82)" means the data clearly preferred 4 over 3
and 5.  It is NOT an accuracy guarantee; see ``README.md``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.dsp import EPS, cmvn, energy_vad, mfcc, stft
from ..core.types import AudioBuffer
from ..core.utils import module_available

DEFAULT_MAX_K = 8


# --------------------------------------------------------------------------- #
#  windowing + features
# --------------------------------------------------------------------------- #
def speech_windows(
    audio: AudioBuffer,
    win_s: float = 1.5,
    hop_s: float = 0.75,
    min_speech_frac: float = 0.6,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Return ([(start, end)], mask_of_speech) for windows that contain speech."""
    mask, hop = energy_vad(audio.samples, audio.sr)
    if len(mask) == 0:
        return [], mask
    win_frames = max(1, int(win_s / hop))
    hop_frames = max(1, int(hop_s / hop))

    out: List[Tuple[float, float]] = []
    for start in range(0, max(1, len(mask) - win_frames + 1), hop_frames):
        chunk = mask[start: start + win_frames]
        if chunk.size and chunk.mean() >= min_speech_frac:
            out.append((start * hop, (start + win_frames) * hop))
    return out, mask


#: how strongly pitch is weighted relative to a single MFCC dimension.
#: f0 is by far the most speaker-discriminative scalar we have, so without this
#: it gets drowned out by the 20 cepstral dimensions (measurably worse K).
F0_WEIGHT = 3.0
FEATURE_DIM = 12 + 7 + 3  # mfcc means (c1..c12), mfcc stds (c1..c7), f0/slope/centroid


def window_features(audio: AudioBuffer, windows: List[Tuple[float, float]]) -> np.ndarray:
    """Numpy fallback embedding: MFCC statistics + pitch + spectral shape.

    ``c0`` (overall loudness) is deliberately dropped -- it tracks how close the
    person sat to the mic, not who they are, and it makes clusters split on
    volume instead of on voice.
    """
    feats = []
    sr = audio.sr
    for start, end in windows:
        seg = audio.samples[int(start * sr): int(end * sr)]
        if seg.size < sr // 8:
            feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue
        m = cmvn(mfcc(seg, sr, n_mfcc=20, n_fft=512, hop=160))
        vec = np.concatenate(
            [
                m[1:13].mean(axis=1),                  # 12  timbre
                m[1:8].std(axis=1),                    # 7   timbre variability
                [F0_WEIGHT * _f0_estimate(seg, sr)],   # 1   pitch (strongest cue)
                [_spectral_slope(seg, sr)],            # 1   voice "darkness"
                [_spectral_centroid(seg, sr)],         # 1
            ]
        ).astype(np.float32)
        feats.append(vec)
    arr = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return _standardise(arr)


def _standardise(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _f0_estimate(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Autocorrelation pitch in semitones relative to 100 Hz (0 if unvoiced)."""
    x = x - x.mean()
    if x.size < sr // 20:
        return 0.0
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(corr) - 1)
    if hi <= lo:
        return 0.0
    peak = int(np.argmax(corr[lo:hi])) + lo
    if corr[peak] <= 0.25 * (corr[0] + EPS):
        return 0.0
    f0 = sr / float(peak)
    return float(12.0 * np.log2(max(f0, 1e-3) / 100.0))


def _spectral_slope(x: np.ndarray, sr: int) -> float:
    spec = np.abs(stft(x, n_fft=512, hop=256)).mean(axis=1)
    freqs = np.fft.rfftfreq(512, 1.0 / sr)
    log_mag = np.log(np.maximum(spec, EPS))
    if len(freqs) < 3:
        return 0.0
    slope = np.polyfit(freqs, log_mag, 1)[0]
    return float(slope * 1000.0)


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    spec = np.abs(stft(x, n_fft=512, hop=256)).mean(axis=1)
    freqs = np.fft.rfftfreq(512, 1.0 / sr)
    total = float(spec.sum()) + EPS
    return float(np.sum(freqs * spec) / total / 1000.0)


# --------------------------------------------------------------------------- #
#  optional: real speaker embeddings
# --------------------------------------------------------------------------- #
def ecapa_embeddings(audio: AudioBuffer, windows: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """SpeechBrain ECAPA-TDNN embeddings (192-d) if speechbrain + torch exist."""
    if not (module_available("torch") and module_available("speechbrain")):
        return None
    try:
        import torch  # type: ignore

        from ..core.audio_io import resample

        try:  # speechbrain >= 1.0
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except Exception:  # speechbrain 0.5
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        from ..core.utils import PRETRAINED_DIR

        kwargs = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "savedir": str(PRETRAINED_DIR / "ecapa"),
            "run_opts": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        }
        # Copy rather than symlink out of the HF cache -- symlinks need
        # Developer Mode on Windows (WinError 1314).  See methods/sepformer.py.
        try:
            from speechbrain.utils.fetching import LocalStrategy  # type: ignore

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:  # speechbrain 0.5.x
            pass

        try:
            model = EncoderClassifier.from_hparams(**kwargs)
        except TypeError:  # older signature
            kwargs.pop("local_strategy", None)
            model = EncoderClassifier.from_hparams(**kwargs)
        wav16 = resample(audio, 16000)
        chunks = []
        for start, end in windows:
            seg = wav16.samples[int(start * 16000): int(end * 16000)]
            if seg.size < 16000 // 2:
                seg = np.pad(seg, (0, 16000 // 2 - seg.size))
            chunks.append(seg)
        if not chunks:
            return None
        n = max(len(c) for c in chunks)
        batch = np.stack([np.pad(c, (0, n - len(c))) for c in chunks])
        with torch.no_grad():
            emb = model.encode_batch(torch.from_numpy(batch).float()).squeeze(1).cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + EPS)
        return emb.astype(np.float32)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  clustering (numpy k-means so sklearn stays optional)
# --------------------------------------------------------------------------- #
def kmeans(x: np.ndarray, k: int, iters: int = 60, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if k >= n:
        return np.arange(n) % k, x.copy()

    # k-means++ seeding
    centres = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((x[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + EPS)
        centres.append(x[rng.choice(n, p=probs)])
    centres = np.asarray(centres, dtype=np.float32)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    return labels, centres


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(x) <= len(uniq):
        return -1.0
    dist = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), 0.0))
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == other].mean() for other in uniq if other != labels[i])
        scores.append((b - a) / max(a, b, EPS))
    return float(np.mean(scores)) if scores else -1.0


def eigengap_k(x: np.ndarray, max_k: int) -> Tuple[int, List[float]]:
    """Spectral-clustering style estimate: the largest gap in the eigenvalues."""
    if len(x) < 4:
        return 1, []
    d = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), 0.0))
    sigma = np.median(d[d > 0]) if (d > 0).any() else 1.0
    aff = np.exp(-(d**2) / (2 * sigma**2 + EPS))
    np.fill_diagonal(aff, 0.0)
    deg = aff.sum(axis=1) + EPS
    lap = np.eye(len(x)) - (aff / np.sqrt(np.outer(deg, deg)))
    vals = np.sort(np.linalg.eigvalsh(lap))[: max_k + 2]
    gaps = np.diff(vals)
    if len(gaps) < 2:
        return 1, [float(v) for v in vals]
    # the first eigenvalue of a connected graph is always ~0, so the 0 -> lambda_2
    # jump is not informative; start the search one step later.
    return int(np.argmax(gaps[1:]) + 2), [float(v) for v in vals]


# --------------------------------------------------------------------------- #
#  the public estimator
# --------------------------------------------------------------------------- #
def estimate_speaker_count(
    audio: AudioBuffer,
    min_k: int = 1,
    max_k: int = DEFAULT_MAX_K,
    single_speaker_threshold: float = 0.14,
    use_embeddings: bool = True,
) -> Dict[str, object]:
    """Return ``{n_speakers, confidence, scores, features, windows, method}``."""
    windows, _ = speech_windows(audio)
    if len(windows) < 3:
        return {
            "n_speakers": 1,
            "confidence": 0.25,
            "scores": {},
            "method": "too-short",
            "n_windows": len(windows),
            "windows": windows,
            "features": np.zeros((0, 1), dtype=np.float32),
        }

    feats = ecapa_embeddings(audio, windows) if use_embeddings else None
    backend = "ecapa-tdnn"
    if feats is None or len(feats) != len(windows):
        feats = window_features(audio, windows)
        backend = "mfcc-numpy"

    max_k = int(max(min_k, min(max_k, len(feats) // 2)))
    scores: Dict[int, float] = {}
    labelings: Dict[int, np.ndarray] = {}
    for k in range(max(2, min_k), max_k + 1):
        labels, _ = kmeans(feats, k, seed=17)
        scores[k] = silhouette(feats, labels)
        labelings[k] = labels

    gap_k, eigenvalues = eigengap_k(feats, max_k)

    if not scores:
        best_k, confidence = 1, 0.3
    else:
        best_k = max(scores, key=lambda kk: scores[kk])
        best = scores[best_k]
        rest = sorted((v for kk, v in scores.items() if kk != best_k), reverse=True)
        runner_up = rest[0] if rest else -1.0
        margin = float(np.clip((best - runner_up) / max(abs(best), 0.1), 0.0, 1.0))
        agreement = 0.15 if gap_k == best_k else 0.0
        confidence = float(np.clip(0.35 + 0.5 * margin + agreement + 0.3 * max(best, 0), 0.05, 0.99))
        if best < single_speaker_threshold:
            best_k, confidence = 1, float(np.clip(0.4 + (single_speaker_threshold - best), 0.3, 0.9))

    return {
        "n_speakers": int(best_k),
        "confidence": round(float(confidence), 4),
        "scores": {str(k): round(float(v), 4) for k, v in scores.items()},
        "eigengap_k": int(gap_k),
        "eigenvalues": [round(v, 4) for v in eigenvalues[:8]],
        "method": backend,
        "n_windows": len(windows),
        "windows": windows,
        "features": feats,
        "labels": labelings.get(best_k, np.zeros(len(feats), dtype=int)),
    }
