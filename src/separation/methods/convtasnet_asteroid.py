"""Method 3 -- Conv-TasNet / DPRNN from the Asteroid model zoo.

Install
-------
    pip install torch torchaudio
    pip install asteroid

Asteroid hosts dozens of ready-to-use separation checkpoints on the HF hub.
Conv-TasNet is the fully-convolutional time-domain workhorse: much faster than
SepFormer (no attention), ~5 M parameters, and still a strong 2-speaker
separator.  It is the model we also re-implement in ``architectures.py`` for the
local training path, so this backend doubles as the quality reference for
"how good should my own model get".

Model zoo (pick with ``ASTEROID_MODEL``)
----------------------------------------
* ``mpariente/ConvTasNet_WHAM!_sepclean``   -- 8 kHz, 2 src, noisy mixtures
* ``JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`` -- 16 kHz, 2 src (recommended)
* ``JorisCos/ConvTasNet_Libri3Mix_sepclean_16k`` -- 16 kHz, 3 src
* ``mpariente/DPRNNTasNet-ks2_WHAM_sepclean`` -- DPRNN variant, better SI-SDR
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseSeparator
from ..chunking import chunked_separate

MODEL_TABLE = {
    "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k": (16000, 2),
    "JorisCos/ConvTasNet_Libri3Mix_sepclean_16k": (16000, 3),
    "mpariente/ConvTasNet_WHAM!_sepclean": (8000, 2),
    "mpariente/DPRNNTasNet-ks2_WHAM_sepclean": (8000, 2),
    "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k": (16000, 1),
}


def pick_model(num_speakers: Optional[int]) -> str:
    override = os.environ.get("ASTEROID_MODEL", "").strip()
    if override:
        return override
    if num_speakers and num_speakers >= 3:
        return "JorisCos/ConvTasNet_Libri3Mix_sepclean_16k"
    return "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"


@register_separator
class AsteroidConvTasNetSeparator(BaseSeparator):
    info = MethodInfo(
        key="convtasnet_asteroid",
        name="Conv-TasNet (Asteroid zoo)",
        kind="separate",
        family="deep-pretrained",
        description=(
            "Time-domain fully-convolutional separator from the Asteroid model zoo. "
            "3-5x faster than SepFormer with most of the quality -- the practical "
            "default for overlapping 2-3 speaker audio."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=True,
        pip=["torch", "asteroid"],
        install_hint="pip install torch torchaudio && pip install asteroid",
        notes="Fixed source count per checkpoint (2 or 3). 16 kHz Libri models recommended.",
        max_speakers=3,
    )
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("asteroid"):
            return False, "missing python package(s): asteroid"
        return True, ""

    def load(self) -> None:
        self._cache: dict = {}
        self._loaded = True

    def _get_model(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
        import torch  # type: ignore
        from asteroid.models import BaseModel  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BaseModel.from_pretrained(model_id).to(device).eval()
        self._cache[model_id] = (model, device)
        return self._cache[model_id]

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        import torch  # type: ignore

        from ...core.audio_io import resample

        model_id = pick_model(num_speakers)
        sr, n_src = MODEL_TABLE.get(model_id, (16000, 2))
        work = audio if audio.sr == sr else resample(audio, sr)
        model, device = self._get_model(model_id)

        def run_block(block: np.ndarray) -> List[np.ndarray]:
            tensor = torch.from_numpy(np.ascontiguousarray(block)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                est = model.separate(tensor) if hasattr(model, "separate") else model(tensor)
            est = est.squeeze(0).cpu().numpy()  # (n_src, time)
            if est.ndim == 1:
                est = est[None, :]
            return [est[i].astype(np.float32) for i in range(est.shape[0])]

        sources = chunked_separate(run_block, work.samples, sr, chunk_s=20.0, overlap_s=1.0)

        return sources, {
            "backend": "asteroid",
            "output_sr": sr,
            "confidence": 0.75,
            "metrics": {
                "model_id": model_id,
                "model_sample_rate": sr,
                "model_max_sources": n_src,
                "device": device,
                "requested_speakers": num_speakers,
            },
        }
