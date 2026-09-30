"""Method 5 -- MossFormer2 via ClearerVoice-Studio (`clearvoice` package).

Install
-------
    pip install torch torchaudio
    pip install clearvoice

MossFormer2 is currently one of the highest scoring open separation models on
WSJ0-2mix (>22 dB SI-SDRi).  It combines a gated single-head transformer with a
recurrent FSMN module, so it keeps long-range context without SepFormer's
quadratic cost.  ClearerVoice-Studio (Alibaba) ships it as a one-line package
together with enhancement and super-resolution models.

Available tasks/models in the package (set ``CLEARVOICE_MODEL``):
* ``MossFormer2_SS_16K``  -- speech separation, 16 kHz, 2 speakers  (default)
* ``MossFormer2_SE_48K``  -- speech *enhancement* (denoise) 48 kHz
* ``MossFormerGAN_SE_16K``-- GAN enhancement 16 kHz
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ...core.audio_io import save_audio
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CACHE_DIR, module_available, new_id
from ..base import BaseSeparator


@register_separator
class MossFormerClearVoiceSeparator(BaseSeparator):
    info = MethodInfo(
        key="mossformer_clearvoice",
        name="MossFormer2 (ClearerVoice)",
        kind="separate",
        family="deep-pretrained",
        description=(
            "Alibaba's MossFormer2 -- gated single-head transformer + FSMN, one of "
            "the top scoring open separators on WSJ0-2mix. Packaged as a one-liner "
            "by ClearerVoice-Studio."
        ),
        speed="slow",
        quality=5,
        needs_gpu=True,
        offline=True,
        pip=["torch", "clearvoice"],
        install_hint="pip install torch torchaudio && pip install clearvoice",
        notes="2 speakers at 16 kHz. Weights (~200 MB) download on first run. GPU strongly advised.",
        max_speakers=2,
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("clearvoice"):
            return False, "missing python package(s): clearvoice"
        return True, ""

    def load(self) -> None:
        from clearvoice import ClearVoice  # type: ignore

        self._model_name = os.environ.get("CLEARVOICE_MODEL", "MossFormer2_SS_16K")
        if "_SS_" not in self._model_name:
            raise ValueError("CLEARVOICE_MODEL must be a speech separation (_SS_) checkpoint")
        self._model = ClearVoice(task="speech_separation", model_names=[self._model_name])
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        tmp = CACHE_DIR / (new_id("clearvoice") + ".wav")
        save_audio(tmp, audio)
        try:
            out = self._model(input_path=str(tmp), online_write=False)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

        sources = self._to_sources(out)
        if not sources:
            raise RuntimeError("clearvoice returned an unexpected payload: %s" % type(out))

        return sources, {
            "backend": "clearvoice/" + self._model_name,
            "output_sr": self.target_sr,
            "confidence": 0.85,
            "metrics": {
                "model": self._model_name,
                "model_max_sources": 2,
                "requested_speakers": num_speakers,
            },
        }

    @staticmethod
    def _to_sources(out) -> List[np.ndarray]:
        """ClearVoice returns ndarray / list / dict depending on the version."""
        while isinstance(out, dict):
            if len(out) != 1:
                raise RuntimeError("Expected one ClearVoice model/input result")
            out = next(iter(out.values()))
        if out is None:
            return []
        # File inference commonly returns [source, channel=1, samples]. The old
        # batch assumption dropped the second speaker from this exact shape.
        arr = np.squeeze(np.asarray(out, dtype=np.float32))
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            # orient as (n_src, time)
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]
        return []
