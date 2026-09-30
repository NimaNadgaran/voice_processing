"""Denoising stage.

Public surface::

    from src.denoising import denoise, available_methods

    result = denoise("recording.wav", method="deepfilternet")
    result.audio      # AudioBuffer
    result.metrics    # estimated SNR gain, noise-floor drop, ...

Read ``src/denoising/README.md`` for what each backend does, when to pick it,
and how long the trainable one takes to train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

from ..core.audio_io import load_audio, save_audio
from ..core.registry import get_denoiser, list_denoisers
from ..core.types import AudioBuffer, DenoiseResult, MethodInfo
from .base import BaseDenoiser, DenoiserUnavailable

DEFAULT_METHOD = "spectral_gate"


def available_methods(only_available: bool = False) -> List[MethodInfo]:
    return list_denoisers(only_available=only_available)


def denoise(
    source: Union[str, Path, AudioBuffer],
    method: str = DEFAULT_METHOD,
    progress: Optional[Callable[[float, str], None]] = None,
    output_path: Optional[Union[str, Path]] = None,
    safe: bool = True,
) -> DenoiseResult:
    """Denoise a file or an :class:`AudioBuffer` with the chosen backend."""
    audio = source if isinstance(source, AudioBuffer) else load_audio(source)
    engine: BaseDenoiser = get_denoiser(method)
    result = engine.safe_run(audio, progress) if safe else engine.run(audio, progress)
    if output_path:
        save_audio(output_path, result.audio)
        result.path = str(output_path)
    return result


__all__ = [
    "denoise",
    "available_methods",
    "BaseDenoiser",
    "DenoiserUnavailable",
    "DEFAULT_METHOD",
]
