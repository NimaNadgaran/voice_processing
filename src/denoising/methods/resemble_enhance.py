"""Method 7 -- Resemble Enhance (denoise + *generative* restoration).

Install
-------
    pip install torch torchaudio
    pip install resemble-enhance

Two stages: a UNet denoiser, then a conditional latent diffusion "enhancer"
that regenerates a clean 44.1 kHz waveform.  Unlike every other method here it
is allowed to *invent* detail, so it can repair clipped, band-limited or
heavily codec-damaged speech that mask-based methods cannot.

Caveat, stated plainly: because it is generative it can subtly change timbre.
Do not use it as the input to forensic/biometric work -- use it when a human
has to listen.  Very slow on CPU (several x real time); a GPU is strongly
recommended.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser


@register_denoiser
class ResembleEnhanceDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="resemble_enhance",
        name="Resemble Enhance (generative)",
        kind="denoise",
        family="deep-pretrained",
        description=(
            "UNet denoiser followed by a latent-diffusion enhancer that regenerates "
            "44.1 kHz speech. Repairs clipping, band-limiting and codec damage that "
            "masking methods physically cannot recover."
        ),
        speed="slow",
        quality=5,
        needs_gpu=True,
        offline=True,
        pip=["torch", "resemble_enhance"],
        install_hint="pip install torch torchaudio && pip install resemble-enhance",
        notes="Generative: may alter timbre. Not for forensic use. GPU strongly advised.",
    )
    target_sr = 44100
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("resemble_enhance"):
            return False, "missing python package(s): resemble-enhance"
        return True, ""

    def load(self) -> None:
        import torch  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        import torch  # type: ignore
        from resemble_enhance.enhancer.inference import denoise, enhance  # type: ignore

        wav = torch.from_numpy(np.ascontiguousarray(audio.samples)).float()

        with torch.no_grad():
            dwav, new_sr = denoise(wav, audio.sr, self._device)
            try:
                dwav, new_sr = enhance(
                    dwav, new_sr, self._device, nfe=32, solver="midpoint", lambd=0.9, tau=0.5
                )
                stage = "denoise+enhance"
            except Exception:
                stage = "denoise-only"

        arr = dwav.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return AudioBuffer(arr, int(new_sr)), {
            "backend": "resemble-enhance",
            "stage": stage,
            "device": self._device,
            "generative": True,
        }
