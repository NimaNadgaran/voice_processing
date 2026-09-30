"""Method 0 -- bypass. Keeps the audio as a single track.

The mirror image of the "no denoising" control: it lets you build a path that
only cleans the audio and hands it back in one piece, with no attempt to split
speakers.  Useful when the recording is one person, when you only want the
denoiser's output, or as the control condition for "did separating actually
help?".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ..base import BaseSeparator


@register_separator
class PassthroughSeparator(BaseSeparator):
    info = MethodInfo(
        key="none",
        name="No separation (single track)",
        kind="separate",
        family="dsp",
        description="Bypass. Keeps the audio whole instead of splitting speakers.",
        speed="realtime",
        quality=1,
        needs_gpu=False,
        offline=True,
        pip=[],
        install_hint="",
        notes="Pair it with a denoiser when you only want a cleaned-up file.",
        max_speakers=1,
    )

    def check_available(self) -> Tuple[bool, str]:
        return True, ""

    def _separate(
        self, audio: AudioBuffer, num_speakers: Optional[int]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        return [audio.samples.copy()], {
            "backend": "passthrough",
            "labels": ["Full mix"],
            "confidence": 1.0,
        }
