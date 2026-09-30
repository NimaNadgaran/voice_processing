"""Speaker separation stage.

Public surface::

    from src.separation import separate, count_speakers, available_methods

    est = count_speakers(audio)            # -> {"n_speakers": 4, "confidence": .82}
    res = separate(audio, method="pyannote", num_speakers=est["n_speakers"])
    for track in res.tracks:               # one entry per speaker
        track.audio, track.segments, track.total_speech

Read ``src/separation/README.md`` for what each backend does, which ones handle
overlapping speech, and how long the trainable one takes to train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ..core.audio_io import load_audio, save_audio
from ..core.registry import get_separator, list_separators
from ..core.types import AudioBuffer, MethodInfo, SeparationResult
from .base import BaseSeparator, SeparatorUnavailable
from .speaker_count import estimate_speaker_count

DEFAULT_METHOD = "diarize_cluster"


def available_methods(only_available: bool = False) -> List[MethodInfo]:
    return list_separators(only_available=only_available)


def count_speakers(
    source: Union[str, Path, AudioBuffer],
    max_speakers: int = 8,
) -> Dict[str, object]:
    audio = source if isinstance(source, AudioBuffer) else load_audio(source)
    out = estimate_speaker_count(audio, max_k=max_speakers)
    out.pop("features", None)  # numpy array -- not json friendly
    out.pop("labels", None)
    out.pop("windows", None)
    return out


def separate(
    source: Union[str, Path, AudioBuffer],
    method: str = DEFAULT_METHOD,
    num_speakers: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "speaker",
    safe: bool = True,
) -> SeparationResult:
    """Split a recording into one track per speaker."""
    audio = source if isinstance(source, AudioBuffer) else load_audio(source)
    engine: BaseSeparator = get_separator(method)
    result = (
        engine.safe_run(audio, num_speakers, progress)
        if safe
        else engine.run(audio, num_speakers, progress)
    )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for track in result.tracks:
            path = out_dir / ("%s_%02d.wav" % (prefix, track.index + 1))
            save_audio(path, track.audio)
            track.path = str(path)
    return result


__all__ = [
    "separate",
    "count_speakers",
    "available_methods",
    "BaseSeparator",
    "SeparatorUnavailable",
    "DEFAULT_METHOD",
]
