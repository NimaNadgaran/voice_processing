"""Dataset for training the local separator.

Mixtures are generated **on the fly** from clean speech, WSJ0-mix style:

1. pick ``n_src`` *different speakers*,
2. take a random crop from one file of each,
3. scale each to a random relative level (default +-2.5 dB),
4. sum them -> that is the network input; the individual crops are the targets.

Speaker identity matters: mixing two crops of the *same* speaker teaches the
model nothing (the permutation is ambiguous), so the loader groups files by
speaker.

Expected layout (any of these works)
------------------------------------
    data/datasets/speakers/<speaker_id>/*.wav      <- best, explicit
    LibriSpeech/train-clean-100/<spk>/<chapter>/*.flac   <- auto-detected
    a flat folder of files                          <- falls back to "one
                                                       speaker per file" with
                                                       a printed warning

Optionally add noise (``noise_dir``) to train a *noisy* separator, which is what
you want if the audio is never going to be studio clean -- that is the WHAM!
recipe and it transfers much better to real recordings.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


# --------------------------------------------------------------------------- #
def scan_audio(root: "str | Path") -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob("*" + ext))
    return sorted(f for f in files if f.is_file() and f.stat().st_size > 2048)


def group_by_speaker(root: "str | Path", min_files: int = 1) -> Dict[str, List[Path]]:
    """Map speaker id -> files.

    Heuristics, in order:
    * ``<root>/<speaker>/...``            (the parent-most directory under root)
    * LibriSpeech ``<spk>/<chapter>/``  (grandparent when the parent is numeric)
    * flat folder                       -> every file is its own "speaker"
    """
    root = Path(root)
    files = scan_audio(root)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = Path(f.name)
        parts = rel.parts
        if len(parts) >= 3 and parts[1].isdigit():      # LibriSpeech style
            speaker = parts[0]
        elif len(parts) >= 2:
            speaker = parts[0]
        else:
            speaker = f.stem                            # flat folder
        groups[speaker].append(f)
    return {k: v for k, v in groups.items() if len(v) >= min_files}


def _read(path: Path, sr: int) -> np.ndarray:
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


def _active_crop(x: np.ndarray, n: int, rng: random.Random, tries: int = 6) -> np.ndarray:
    """Random crop that is not silence (silent targets make PIT unstable)."""
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
    best, best_energy = None, -1.0
    for _ in range(tries):
        start = rng.randint(0, max(0, len(x) - n))
        seg = x[start: start + n]
        energy = float(np.mean(seg**2))
        if energy > best_energy:
            best, best_energy = seg, energy
        if energy > 1e-4:
            return seg
    return best if best is not None else x[:n]


# --------------------------------------------------------------------------- #
class SeparationDataset(Dataset):
    """Returns ``(mixture (T,), sources (n_src, T))``."""

    def __init__(
        self,
        clean_dir: "str | Path",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment_seconds: float = 4.0,
        level_range_db: Tuple[float, float] = (-2.5, 2.5),
        noise_dir: Optional["str | Path"] = None,
        noise_snr_range: Tuple[float, float] = (10.0, 30.0),
        length: int = 4000,
        seed: int = 0,
        cache_size: int = 96,
    ) -> None:
        self.sr = int(sample_rate)
        self.n = int(segment_seconds * sample_rate)
        self.n_src = int(n_src)
        self.level_range_db = level_range_db
        self.noise_snr_range = noise_snr_range
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

        self.speakers = group_by_speaker(clean_dir)
        self.speaker_ids = sorted(self.speakers)
        if len(self.speaker_ids) < self.n_src:
            raise ValueError(
                "found %d speaker folder(s) under %s but n_src=%d. Use one folder "
                "per speaker -- see the docstring." % (len(self.speaker_ids), clean_dir, self.n_src)
            )
        if len(self.speaker_ids) < 20:
            print("  ! only %d speakers found -- the model will overfit to these voices."
                  % len(self.speaker_ids))

        self.noise_files = scan_audio(noise_dir) if noise_dir else []
        self._cache: dict = {}
        self._cache_size = cache_size

    def __len__(self) -> int:
        return self.length

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

        chosen = rng.sample(self.speaker_ids, self.n_src)
        sources = []
        for speaker in chosen:
            audio = self._load(rng.choice(self.speakers[speaker]))
            seg = _active_crop(audio, self.n, rng).astype(np.float32).copy()
            seg -= float(np.mean(seg))
            rms = float(np.sqrt(np.mean(seg**2))) + 1e-9
            gain = 10 ** (rng.uniform(*self.level_range_db) / 20.0)
            sources.append(seg / rms * 0.05 * gain)

        stack = np.stack(sources).astype(np.float32)
        mixture = stack.sum(axis=0)

        if self.noise_files and rng.random() < 0.5:
            noise = _active_crop(self._load(rng.choice(self.noise_files)), self.n, rng)
            snr = rng.uniform(*self.noise_snr_range)
            mix_p = float(np.mean(mixture**2)) + 1e-12
            noise_p = float(np.mean(noise**2)) + 1e-12
            mixture = mixture + noise * math.sqrt(mix_p / (noise_p * 10 ** (snr / 10.0)))

        peak = float(np.max(np.abs(mixture))) or 1.0
        if peak > 0.99:
            mixture = mixture / peak * 0.99
            stack = stack / peak * 0.99

        if torch is None:
            return mixture.astype(np.float32), stack.astype(np.float32)
        return torch.from_numpy(mixture.astype(np.float32)), torch.from_numpy(stack.astype(np.float32))


# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: dict):
    from torch.utils.data import DataLoader

    common = dict(
        clean_dir=cfg["clean_dir"],
        n_src=cfg.get("n_src", 2),
        sample_rate=cfg.get("sample_rate", 8000),
        segment_seconds=cfg.get("segment_seconds", 4.0),
        noise_dir=cfg.get("noise_dir"),
    )
    train = SeparationDataset(
        length=cfg.get("steps_per_epoch", 1000) * cfg.get("batch_size", 4),
        seed=cfg.get("seed", 0), **common,
    )
    val = SeparationDataset(length=cfg.get("val_items", 150), seed=987_654, **common)
    from ...core.training_split import held_out
    train.speaker_ids, val.speaker_ids = held_out(
        train.speaker_ids, cfg.get("validation_fraction", 0.2),
        cfg.get("seed", 0), minimum=train.n_src,
    )
    train.speakers = {key: train.speakers[key] for key in train.speaker_ids}
    val.speakers = {key: val.speakers[key] for key in val.speaker_ids}
    if train.noise_files:
        train.noise_files, val.noise_files = held_out(
            train.noise_files, cfg.get("validation_fraction", 0.2), cfg.get("seed", 0)
        )

    workers = int(cfg.get("num_workers", 0))
    return (
        DataLoader(train, batch_size=cfg.get("batch_size", 4), shuffle=False,
                   num_workers=workers, drop_last=True),
        DataLoader(val, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=0),
    )
