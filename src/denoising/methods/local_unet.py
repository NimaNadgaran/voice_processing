"""Method 8 -- **our own** locally trained denoiser (SpectralUNet).

This is the "train it yourself" path.  Nothing is downloaded: the weights come
from ``models/checkpoints/denoise_unet_best.pt``, produced by

    python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml

Until that file exists the method reports itself as *unavailable* in the UI with
the exact command to produce it (no training is started automatically -- ever).

Why bother when DeepFilterNet exists?
-------------------------------------
* it is *yours*: you can fine-tune it on your own room, mic and noise types,
  which beats any generic model on that specific domain;
* it is fully offline and dependency-light at inference time;
* it is a working reference implementation to learn from / extend.

See ``src/denoising/README.md`` for the training recipe and time estimates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CKPT_DIR, module_available
from ..base import BaseDenoiser
from ..chunking import chunked_enhance

DEFAULT_CKPT = CKPT_DIR / "denoise_unet_best.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("DENOISE_UNET_CKPT", str(DEFAULT_CKPT)))


@register_denoiser
class LocalUNetDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="local_unet",
        name="Local SpectralUNet (your model)",
        kind="denoise",
        family="deep-local",
        description=(
            "A masking U-Net + GRU trained by you on your own data. Fully offline, "
            "~1.6 M parameters, and tunable to your specific microphone and room."
        ),
        speed="fast",
        quality=4,
        needs_gpu=False,
        offline=True,
        pip=["torch"],
        install_hint=(
            "pip install -r requirements-train.txt, then: "
            "python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml"
        ),
        notes="Needs models/checkpoints/denoise_unet_best.pt. Training is never started automatically.",
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        ckpt = checkpoint_path()
        if not ckpt.exists():
            return False, "no trained checkpoint at %s -- train it first (see install hint)" % ckpt
        return True, ""

    def load(self) -> None:
        import torch  # type: ignore

        from ..architectures import build_model

        ckpt = torch.load(str(checkpoint_path()), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        model = build_model(cfg)
        model.load_state_dict(state)
        model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(self._device)
        self.target_sr = int(cfg.get("sample_rate", 16000))
        self._meta = {
            "epochs_trained": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "val_si_sdr": ckpt.get("val_si_sdr") if isinstance(ckpt, dict) else None,
            "params_m": round(model.n_params / 1e6, 2),
        }
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        import torch  # type: ignore

        x = np.ascontiguousarray(audio.samples)
        sr = audio.sr
        def process(seg):
            if len(seg) < self._model.n_fft:
                seg = np.pad(seg, (0, self._model.n_fft - len(seg)))
            tensor = torch.from_numpy(np.ascontiguousarray(seg)).float().unsqueeze(0).to(self._device)
            with torch.inference_mode():
                return self._model(tensor)[0].cpu().numpy()
        out = chunked_enhance(process, x, sr)
        info = {"backend": "local-spectral-unet", "device": self._device,
                "model_sample_rate": sr, "speech_bandwidth_hz": sr // 2}
        info.update({k: v for k, v in getattr(self, "_meta", {}).items() if v is not None})
        return AudioBuffer(out, sr), info
