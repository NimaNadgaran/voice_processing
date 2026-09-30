"""Method 2 -- SepFormer (SpeechBrain), true overlapping-speech separation.

Install
-------
    pip install torch torchaudio
    pip install speechbrain

Weights download from the HF hub on first use into ``models/pretrained/``.

SepFormer is a dual-path transformer that operates on a learned encoder basis
and is trained with permutation-invariant SI-SNR.  Unlike diarization it *does*
split simultaneous speech -- the classic "cocktail party" solution.

Model zoo (pick with ``SEPFORMER_MODEL``)
-----------------------------------------
=============================== ====== ====== ===========================
model id                        sr     #src   trained on
=============================== ====== ====== ===========================
speechbrain/sepformer-wsj02mix  8 kHz  2      WSJ0-2mix (clean, anechoic)
speechbrain/sepformer-wsj03mix  8 kHz  3      WSJ0-3mix
speechbrain/sepformer-whamr16k  16 kHz 2      WHAMR! (noisy + reverberant)
speechbrain/resepformer-wsj02mix 8 kHz 2      lighter/faster variant
=============================== ====== ====== ===========================

Honest limitations
------------------
* The number of sources is baked into the weights (2 or 3). For 4+ speakers use
  diarization, pyannote, or diarization *then* SepFormer per overlapped region.
* WSJ0/Libri models are trained on clean anechoic mixtures -- on a real phone
  recording ``whamr16k`` usually wins. Denoise first (that is what the pipeline
  paths do) and quality jumps.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import PRETRAINED_DIR, module_available
from ..base import BaseSeparator
from ..chunking import chunked_separate

MODEL_TABLE = {
    "speechbrain/sepformer-wsj02mix": (8000, 2),
    "speechbrain/sepformer-wsj03mix": (8000, 3),
    "speechbrain/sepformer-whamr16k": (16000, 2),
    "speechbrain/sepformer-wham16k-enhancement": (16000, 1),
    "speechbrain/resepformer-wsj02mix": (8000, 2),
    "speechbrain/sepformer-libri2mix": (8000, 2),
    "speechbrain/sepformer-libri3mix": (8000, 3),
}


def pick_model(num_speakers: Optional[int]) -> str:
    override = os.environ.get("SEPFORMER_MODEL", "").strip()
    if override:
        return override
    if num_speakers and num_speakers >= 3:
        return "speechbrain/sepformer-wsj03mix"
    return "speechbrain/sepformer-whamr16k"  # noisy+reverberant = closest to real life


@register_separator
class SepFormerSeparator(BaseSeparator):
    info = MethodInfo(
        key="sepformer",
        name="SepFormer (SpeechBrain)",
        kind="separate",
        family="deep-pretrained",
        description=(
            "Dual-path transformer trained with permutation-invariant SI-SNR. "
            "Genuinely splits people talking at the same time -- the cocktail-party "
            "solver. Auto-selects the 2-source or 3-source checkpoint."
        ),
        speed="slow",
        quality=5,
        needs_gpu=False,
        offline=True,
        pip=["torch", "speechbrain"],
        install_hint="pip install torch torchaudio && pip install speechbrain",
        notes="Fixed 2 or 3 sources. Long files are processed in overlapping blocks with permutation stitching.",
        max_speakers=3,
    )
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("speechbrain"):
            return False, "missing python package(s): speechbrain"
        return True, ""

    def load(self) -> None:
        self._cache: dict = {}
        self._loaded = True

    def _get_model(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
        import torch  # type: ignore

        try:  # speechbrain >= 1.0
            from speechbrain.inference.separation import SepformerSeparation  # type: ignore
        except Exception:  # speechbrain 0.5.x
            from speechbrain.pretrained import SepformerSeparation  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {
            "source": model_id,
            "savedir": str(PRETRAINED_DIR / model_id.replace("/", "__")),
            "run_opts": {"device": device},
        }

        # speechbrain >= 1.0 links the HF cache into savedir with a *symlink* by
        # default.  On Windows that needs Developer Mode or an elevated shell,
        # otherwise it dies with WinError 1314 ("a required privilege is not
        # held by the client") after the model has already downloaded.  Copying
        # costs a little disk and always works.
        try:
            from speechbrain.utils.fetching import LocalStrategy  # type: ignore

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:  # speechbrain 0.5.x has no strategies
            pass

        try:
            model = SepformerSeparation.from_hparams(**kwargs)
        except TypeError:  # older signature: no local_strategy keyword
            kwargs.pop("local_strategy", None)
            model = SepformerSeparation.from_hparams(**kwargs)
        self._cache[model_id] = (model, device)
        return self._cache[model_id]

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        import torch  # type: ignore

        model_id = pick_model(num_speakers)
        sr, n_src = MODEL_TABLE.get(model_id, (8000, 2))

        from ...core.audio_io import resample

        work = audio if audio.sr == sr else resample(audio, sr)
        model, device = self._get_model(model_id)
        # Use the checkpoint's own metadata for custom model IDs too.
        sr = int(getattr(getattr(model, 'hparams', None), 'sample_rate', sr))
        n_src = int(getattr(getattr(model, 'hparams', None), 'num_spks', n_src))
        work = audio if audio.sr == sr else resample(audio, sr)

        def run_block(block: np.ndarray) -> List[np.ndarray]:
            if len(block) < 256:
                block = np.pad(block, (0, 256 - len(block)))
            tensor = torch.from_numpy(np.ascontiguousarray(block)).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                est = model.separate_batch(tensor)  # (batch, time, n_src)
            est = est[0].cpu().numpy()
            return [est[:, i].astype(np.float32) for i in range(est.shape[-1])]

        sources = chunked_separate(run_block, work.samples, sr, chunk_s=10.0, overlap_s=1.0)

        return sources, {
            "backend": "sepformer",
            "output_sr": sr,  # the base class resamples back to the input rate
            "confidence": 0.8,
            "metrics": {
                "model_id": model_id,
                "model_sample_rate": sr,
                "model_max_sources": n_src,
                "device": device,
                "requested_speakers": num_speakers,
                "note": ("model outputs a fixed %d sources; extra/empty stems are dropped" % n_src),
            },
        }
