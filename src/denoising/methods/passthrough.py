"""Method 0 -- bypass. Keeps the audio exactly as uploaded.

Not a joke method: it is the control condition.  Comparing "raw -> separation"
against "denoised -> separation" is the fastest way to see whether a denoiser is
actually helping the separator or eating the quiet speaker.
"""

from __future__ import annotations

from typing import Tuple

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ..base import BaseDenoiser


@register_denoiser
class PassthroughDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="none",
        name="No denoising (control)",
        kind="denoise",
        family="dsp",
        description="Bypass. Use it as the A/B control when comparing pipelines.",
        speed="realtime",
        quality=1,
        needs_gpu=False,
        offline=True,
        pip=[],
        install_hint="",
        notes="Costs nothing and proves whether denoising helped or hurt separation.",
    )

    def check_available(self) -> Tuple[bool, str]:
        return True, ""

    def _denoise(self, audio: AudioBuffer):
        return audio.copy(), {"backend": "passthrough"}
