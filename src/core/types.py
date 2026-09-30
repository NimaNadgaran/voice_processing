"""Plain dataclasses shared by every module.

Keeping these dependency-free (numpy only) means the frontend, the pipeline and
the individual backends all speak exactly the same language.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  audio container
# --------------------------------------------------------------------------- #
@dataclass
class AudioBuffer:
    """Mono float32 PCM in [-1, 1] plus its sample rate."""

    samples: np.ndarray
    sr: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim > 1:  # (channels, n) or (n, channels) -> mono
            axis = 0 if arr.shape[0] < arr.shape[-1] else -1
            arr = arr.mean(axis=axis)
        self.samples = np.ascontiguousarray(arr, dtype=np.float32)
        self.sr = int(self.sr)

    # -- convenience ------------------------------------------------------- #
    @property
    def duration(self) -> float:
        return float(len(self.samples)) / float(self.sr) if self.sr else 0.0

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(self.samples.copy(), self.sr)

    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples**2))) if self.samples.size else 0.0


# --------------------------------------------------------------------------- #
#  method description (this is what the frontend renders in the method picker)
# --------------------------------------------------------------------------- #
@dataclass
class MethodInfo:
    key: str  # stable id, e.g. "deepfilternet"
    name: str  # human label
    kind: str  # "denoise" | "separate" | "transcribe"
    family: str  # "dsp" | "deep-local" | "deep-pretrained" | "api"
    description: str
    speed: str = "medium"  # "realtime" | "fast" | "medium" | "slow"
    quality: int = 3  # 1..5, rendered as dots in the UI
    needs_gpu: bool = False
    offline: bool = True  # False => hits the network / a hosted API
    pip: List[str] = field(default_factory=list)
    install_hint: str = ""
    notes: str = ""
    max_speakers: Optional[int] = None  # separators only (None = unlimited)
    available: bool = False  # filled at runtime
    unavailable_reason: str = ""

    # ---- long-form explanation, shown in the UI's "How this works" panel ---
    # Filled in from the stage's explanations.py; see BaseDenoiser.describe().
    how_it_works: str = ""  # a paragraph: what the method actually does
    steps: List[str] = field(default_factory=list)  # the algorithm, step by step
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)  # stated honestly
    reference: str = ""  # paper / repo it comes from
    latency: str = ""  # rough cost, in plain words
    supported_languages: List[str] = field(default_factory=list)  # speech-to-text only
    recommended_languages: List[str] = field(default_factory=list)
    supports_auto_language: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
#  results
# --------------------------------------------------------------------------- #
@dataclass
class DenoiseResult:
    audio: AudioBuffer
    method: str
    elapsed: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    backend_used: str = ""  # e.g. "noisereduce" vs "builtin-fallback"
    path: str = ""


@dataclass
class SpeakerTrack:
    """One separated speaker."""

    index: int
    audio: AudioBuffer
    label: str = ""
    speech_ratio: float = 0.0  # fraction of the track that is speech
    total_speech: float = 0.0  # seconds of speech
    segments: List[List[float]] = field(default_factory=list)  # [[start, end], ...]
    energy_db: float = -120.0
    confidence: float = 0.0
    path: str = ""  # filled once written to disk

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label or f"Speaker {self.index + 1}",
            "speech_ratio": round(float(self.speech_ratio), 4),
            "total_speech": round(float(self.total_speech), 3),
            "segments": [[round(float(s), 3), round(float(e), 3)] for s, e in self.segments],
            "energy_db": round(float(self.energy_db), 2),
            "confidence": round(float(self.confidence), 4),
            "duration": round(self.audio.duration, 3),
            "path": self.path,
        }


@dataclass
class SeparationResult:
    tracks: List[SpeakerTrack]
    method: str
    elapsed: float = 0.0
    n_speakers: int = 0
    n_speakers_confidence: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    backend_used: str = ""

    def __post_init__(self) -> None:
        if not self.n_speakers:
            self.n_speakers = len(self.tracks)
