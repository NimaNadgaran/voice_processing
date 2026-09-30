"""Dataset for training the local denoiser.

Mixtures are built **on the fly**: a clean speech segment + a noise segment at a
random SNR.  That means you never store a noisy corpus, and every epoch sees new
combinations, which is worth several dB compared to a fixed pre-mixed set.

Layout expected
---------------
    data/datasets/clean/**/*.wav      any folder tree of clean speech
    data/datasets/noise/**/*.wav      any folder tree of noise (optional)

If ``noise_dir`` is missing or empty the loader falls back to **synthetic
noise** (white / pink / brown / 50-60 Hz hum / babble made from other clean
files), so you can start training with nothing but speech.

Recommended free corpora -- see ``data/README.md`` for the download commands:
* clean : LibriSpeech (960 h), VoiceBank-DEMAND clean set, Common Voice
* noise : DEMAND, MUSAN, FSDnoisy18k, WHAM! noise, ESC-50
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is only needed for training
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


# --------------------------------------------------------------------------- #
def scan_audio(root: "str | Path", exts: Sequence[str] = AUDIO_EXTS) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob("*" + ext))
    return sorted(f for f in files if f.is_file() and f.stat().st_size > 2048)


def _read(path: Path, sr: int) -> np.ndarray:
    """Read + resample without pulling in the app's audio stack."""
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        try:
            from scipy.signal import resample_poly

            g = math.gcd(int(file_sr), int(sr))
            data = resample_poly(data, sr // g, file_sr // g).astype(np.float32)
        except Exception:
            idx = np.linspace(0, len(data) - 1, int(len(data) * sr / file_sr))
            data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def _random_crop(x: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(x) < n:
        reps = int(np.ceil(n / max(len(x), 1)))
        x = np.tile(x, reps)
    start = rng.randint(0, max(0, len(x) - n))
    return x[start: start + n]


# --------------------------------------------------------------------------- #
#  synthetic noise (so training works with zero noise downloads)
# --------------------------------------------------------------------------- #
def synth_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.integers(0, 5)
    if kind == 0:  # white
        return rng.standard_normal(n).astype(np.float32)
    if kind == 1:  # pink (1/f)
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.maximum(np.fft.rfftfreq(n, 1.0 / sr), 1.0)
        return np.fft.irfft(spec / np.sqrt(freqs), n=n).astype(np.float32)
    if kind == 2:  # brown (1/f^2) -- rumble
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.maximum(np.fft.rfftfreq(n, 1.0 / sr), 1.0)
        return np.fft.irfft(spec / freqs, n=n).astype(np.float32)
    if kind == 3:  # mains hum + harmonics
        t = np.arange(n) / sr
        base = float(rng.choice([50.0, 60.0]))
        out = np.zeros(n, dtype=np.float32)
        for h in range(1, 5):
            out += (1.0 / h) * np.sin(2 * np.pi * base * h * t + rng.uniform(0, 6.28))
        return out.astype(np.float32) + 0.1 * rng.standard_normal(n).astype(np.float32)
    # impulsive clicks / keyboard
    out = 0.05 * rng.standard_normal(n).astype(np.float32)
    for _ in range(int(rng.integers(3, 12))):
        at = int(rng.integers(0, max(1, n - 300)))
        out[at: at + 300] += rng.standard_normal(300).astype(np.float32) * float(rng.uniform(1, 4))
    return out


# --------------------------------------------------------------------------- #
class DenoisingDataset(Dataset):
    """(noisy, clean) pairs of ``segment_seconds`` at ``sample_rate``."""

    def __init__(
        self,
        clean_dir: "str | Path",
        noise_dir: Optional["str | Path"] = None,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        length: int = 4000,
        seed: int = 0,
        augment: bool = True,
        cache_size: int = 64,
    ) -> None:
        self.sr = int(sample_rate)
        self.n = int(segment_seconds * sample_rate)
        self.snr_range = snr_range
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0
        self.augment = augment

        self.clean_files = scan_audio(clean_dir)
        if not self.clean_files:
            raise FileNotFoundError(
                "no audio found under %s -- see data/README.md for free corpora" % clean_dir
            )
        self.noise_files = scan_audio(noise_dir) if noise_dir else []
        self._cache: dict = {}
        self._cache_size = cache_size

    def __len__(self) -> int:
        return self.length

    # -- caching keeps the disk quiet when files are reused ---------------- #
    def _load(self, path: Path) -> np.ndarray:
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        data = _read(path, self.sr)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = data
        return data

    def __getitem__(self, index: int):
        index += self.epoch * self.length
        rng = random.Random(self.seed * 1_000_003 + index)
        nrng = np.random.default_rng(self.seed * 7_919 + index)

        clean = _random_crop(self._load(rng.choice(self.clean_files)), self.n, rng)
        clean = clean - float(np.mean(clean))
        peak = float(np.max(np.abs(clean))) or 1.0
        clean = clean / peak * rng.uniform(0.25, 0.9)

        if self.noise_files and rng.random() < 0.85:
            noise = _random_crop(self._load(rng.choice(self.noise_files)), self.n, rng)
        else:
            noise = synth_noise(self.n, self.sr, nrng)

        # optional babble: another talker used as interference
        if self.augment and rng.random() < 0.15 and len(self.clean_files) > 1:
            other = _random_crop(self._load(rng.choice(self.clean_files)), self.n, rng)
            noise = noise + other * rng.uniform(0.3, 1.0)

        snr = rng.uniform(*self.snr_range)
        clean_p = float(np.mean(clean**2)) + 1e-12
        noise_p = float(np.mean(noise**2)) + 1e-12
        noise = noise * math.sqrt(clean_p / (noise_p * (10 ** (snr / 10.0))))

        noisy = clean + noise
        if self.augment:
            if rng.random() < 0.3:  # random EQ tilt: teaches mic invariance
                tilt = rng.uniform(-0.4, 0.4)
                freqs = np.fft.rfftfreq(self.n, 1.0 / self.sr)
                gain = (1.0 + freqs / (self.sr / 2)) ** tilt
                noisy = np.fft.irfft(np.fft.rfft(noisy) * gain, n=self.n).astype(np.float32)
            if rng.random() < 0.1:  # mild clipping
                noisy = np.clip(noisy, -0.85, 0.85)

        scale = max(float(np.max(np.abs(noisy))), 1e-6)
        if scale > 0.99:
            noisy = noisy / scale * 0.99
            clean = clean / scale * 0.99

        if torch is None:
            return noisy.astype(np.float32), clean.astype(np.float32)
        return torch.from_numpy(noisy.astype(np.float32)), torch.from_numpy(clean.astype(np.float32))


# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: dict):
    """Returns (train_loader, val_loader) from a config dict."""
    from torch.utils.data import DataLoader

    common = dict(
        clean_dir=cfg["clean_dir"],
        noise_dir=cfg.get("noise_dir"),
        sample_rate=cfg.get("sample_rate", 16000),
        segment_seconds=cfg.get("segment_seconds", 2.0),
        snr_range=tuple(cfg.get("snr_range", (-5.0, 20.0))),
    )
    train = DenoisingDataset(length=cfg.get("steps_per_epoch", 1000) * cfg.get("batch_size", 8),
                             seed=cfg.get("seed", 0), augment=True, **common)
    val = DenoisingDataset(length=cfg.get("val_items", 200), seed=999_999, augment=False, **common)
    from ...core.training_split import held_out
    train.clean_files, val.clean_files = held_out(
        train.clean_files, cfg.get("validation_fraction", 0.2), cfg.get("seed", 0)
    )
    if train.noise_files:
        train.noise_files, val.noise_files = held_out(
            train.noise_files, cfg.get("validation_fraction", 0.2), cfg.get("seed", 0)
        )

    workers = int(cfg.get("num_workers", 0))
    return (
        DataLoader(train, batch_size=cfg.get("batch_size", 8), shuffle=False,
                   num_workers=workers, pin_memory=False, drop_last=True),
        DataLoader(val, batch_size=cfg.get("batch_size", 8), shuffle=False,
                   num_workers=0, pin_memory=False),
    )
