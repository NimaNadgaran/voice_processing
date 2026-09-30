"""Method 7 -- **our own** locally trained Conv-TasNet separator.

Weights come from ``models/checkpoints/separation_convtasnet_best.pt``, produced by

    python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml

Until that file exists the method is listed as *unavailable* with that command
attached.  Nothing is trained automatically -- ever.

The training script builds its mixtures on the fly from any folder of clean
speech (LibriSpeech, Common Voice, your own recordings), so you can train a
separator specialised to your language, mic and speaker set, which is exactly
where generic checkpoints are weakest.

See ``src/separation/README.md`` for the recipe and the training-time table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CKPT_DIR, module_available
from ..base import BaseSeparator
from ..chunking import chunked_separate

DEFAULT_CKPT = CKPT_DIR / "separation_convtasnet_best.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("SEPARATION_CKPT", str(DEFAULT_CKPT)))


@register_separator
class LocalConvTasNetSeparator(BaseSeparator):
    info = MethodInfo(
        key="local_convtasnet",
        name="Local Conv-TasNet (your model)",
        kind="separate",
        family="deep-local",
        description=(
            "Conv-TasNet trained by you with permutation-invariant SI-SNR on "
            "mixtures generated from your own clean speech. Fully offline and "
            "tunable to your speakers and language."
        ),
        speed="fast",
        quality=4,
        needs_gpu=False,
        offline=True,
        pip=["torch"],
        install_hint=(
            "pip install -r requirements-train.txt, then: "
            "python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml"
        ),
        notes="Needs models/checkpoints/separation_convtasnet_best.pt. Source count is fixed at training time.",
        max_speakers=None,  # whatever n_src the checkpoint was trained with
    )
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
        self._sr = int(cfg.get("sample_rate", 8000))
        self.info.max_speakers = int(cfg.get("n_src", 2))
        self._meta = {
            "epochs_trained": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "val_si_sdr_db": ckpt.get("val_si_sdr") if isinstance(ckpt, dict) else None,
            "params_m": round(model.n_params / 1e6, 2),
            "n_src": int(cfg.get("n_src", 2)),
        }
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        import torch  # type: ignore

        from ...core.audio_io import resample

        work = audio if audio.sr == self._sr else resample(audio, self._sr)

        def run_block(block: np.ndarray) -> List[np.ndarray]:
            tensor = torch.from_numpy(np.ascontiguousarray(block)).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                est = self._model(tensor)[0].cpu().numpy()
            return [est[i].astype(np.float32) for i in range(est.shape[0])]

        sources = chunked_separate(run_block, work.samples, self._sr, chunk_s=20.0, overlap_s=1.0)

        metrics = {"requested_speakers": num_speakers, "model_sample_rate": self._sr}
        metrics.update({k: v for k, v in getattr(self, "_meta", {}).items() if v is not None})
        return sources, {
            "backend": "local-convtasnet",
            "output_sr": self._sr,
            "confidence": 0.7,
            "metrics": metrics,
        }
