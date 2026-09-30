"""Method 5 -- Demucs vocal isolation (music / background removal).

Install
-------
    pip install torch torchaudio
    pip install demucs

Different job from a "denoiser": htdemucs splits a mixture into
``drums / bass / other / vocals``.  Keeping only the **vocals** stem is by far
the best way to rescue speech that sits on top of *music*, TV in the background,
or heavy environmental texture -- cases where DeepFilterNet and spectral gates
struggle because the interference is not noise-like at all.

Reference: Rouard et al., "Hybrid Transformers for Music Source Separation",
ICASSP 2023.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser


@register_denoiser
class DemucsVocalsDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="demucs_vocals",
        name="Demucs Vocal Isolation (htdemucs)",
        kind="denoise",
        family="deep-pretrained",
        description=(
            "Hybrid-Transformer source separation used as a denoiser: throw away "
            "drums/bass/other and keep the vocals stem. The best option when the "
            "background is music or TV rather than hiss."
        ),
        speed="slow",
        quality=5,
        needs_gpu=False,
        offline=True,
        pip=["torch", "demucs"],
        install_hint="pip install torch torchaudio && pip install demucs",
        notes="~80 MB weights on first run. Slow on CPU (roughly 1x real time).",
    )
    target_sr = 44100
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("demucs"):
            return False, "missing python package(s): demucs"
        return True, ""

    def load(self) -> None:
        import torch  # type: ignore
        from demucs.pretrained import get_model  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model("htdemucs")
        model.eval()
        self._model = model.to(self._device)
        self.target_sr = int(getattr(model, "samplerate", 44100))
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        import torch  # type: ignore
        from demucs.apply import apply_model  # type: ignore

        model = self._model
        channels = int(getattr(model, "audio_channels", 2))
        wav = torch.from_numpy(np.ascontiguousarray(audio.samples)).float().unsqueeze(0)
        if channels > 1:
            wav = wav.repeat(channels, 1)

        ref = wav.mean(0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        wav = (wav - mean) / std

        with torch.no_grad():
            sources = apply_model(
                model, wav.unsqueeze(0).to(self._device), split=True, overlap=0.15, progress=False
            )[0]
        sources = sources * std + mean

        names = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
        idx = names.index("vocals") if "vocals" in names else len(names) - 1
        vocals = sources[idx].mean(0).cpu().numpy().astype(np.float32)

        energies = {n: float(np.mean(sources[i].cpu().numpy() ** 2)) for i, n in enumerate(names)}
        total = sum(energies.values()) + 1e-12
        return AudioBuffer(vocals, audio.sr), {
            "backend": "htdemucs",
            "device": self._device,
            "stem_energy_share": {k: round(v / total, 4) for k, v in energies.items()},
        }
