"""Method 4 -- Meta "denoiser" (Demucs waveform speech enhancement, DNS64).

Install
-------
    pip install torch torchaudio
    pip install denoiser

Fully time-domain U-Net (encoder / LSTM / decoder) trained on the DNS challenge
data at 16 kHz.  It is heavier than DeepFilterNet but has a different failure
mode -- it is very good at *non stationary* noise (door slams, clicks, keyboard)
because it never leaves the waveform domain.

Reference: Defossez et al., "Real Time Speech Enhancement in the Waveform
Domain", INTERSPEECH 2020.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser


@register_denoiser
class DemucsDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="demucs_denoiser",
        name="Demucs Denoiser (DNS64)",
        kind="denoise",
        family="deep-pretrained",
        description=(
            "Meta's waveform-domain U-Net + LSTM denoiser. Excellent on sudden, "
            "non-stationary noise (clicks, slams, keyboards) where spectral methods "
            "leave smears."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=True,
        pip=["torch", "denoiser"],
        install_hint="pip install torch torchaudio && pip install denoiser",
        notes="Works at 16 kHz -- anything above 8 kHz is regenerated on the way back.",
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("denoiser"):
            return False, "missing python package(s): denoiser"
        return True, ""

    def load(self) -> None:
        import torch  # type: ignore
        from denoiser import pretrained  # type: ignore

        model = pretrained.dns64()
        model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(self._device)
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        import torch  # type: ignore

        wav = torch.from_numpy(np.ascontiguousarray(audio.samples)).float()
        wav = wav.unsqueeze(0).unsqueeze(0).to(self._device)  # (batch, chan, time)

        # chunk long files so CPU RAM stays bounded (60 s blocks, 0.5 s overlap)
        sr = audio.sr
        block, overlap = 60 * sr, sr // 2
        pieces = []
        with torch.no_grad():
            if wav.shape[-1] <= block:
                pieces.append(self._model(wav)[0, 0].cpu().numpy())
            else:
                pos = 0
                while pos < wav.shape[-1]:
                    end = min(pos + block, wav.shape[-1])
                    chunk = wav[..., max(0, pos - overlap): end]
                    out = self._model(chunk)[0, 0].cpu().numpy()
                    if pos > 0:
                        out = out[overlap:]
                    pieces.append(out)
                    pos = end
        arr = np.concatenate(pieces).astype(np.float32) if pieces else audio.samples
        return AudioBuffer(arr, audio.sr), {"backend": "denoiser-dns64", "device": self._device}
