"""Method 8 -- configured Hugging Face source-separation endpoint.

Setup
-----
1. Free account on huggingface.co -> create a read token.
2. ``set HF_TOKEN=hf_xxx`` (Windows) / ``export HF_TOKEN=hf_xxx`` (Unix).
3. Set HF_SEPARATION_URL to a deployed compatible audio endpoint, or choose an
   HF_SEPARATION_MODEL that actually has hosted inference.

PRIVACY NOTE -- this is the only separator that leaves your machine: the audio
is uploaded to Hugging Face for inference.  It is therefore opt-in (no token =
method disabled) and the UI marks it with an "off-device" badge.

The audio-to-audio pipeline returns one base64 blob per separated source, which
we decode straight into speaker tracks. Endpoints may be rate limited or
cold-start slowly (503 while the model loads -- we retry).

Being on the Hub does not imply hosted inference. A deployed endpoint must
accept WAV uploads and return one JSON audio blob per source.
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional, Tuple

import numpy as np

from ...core.audio_io import load_from_bytes, resample
from ...core.hosted_audio import API_ROOT, request_audio
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseSeparator

DEFAULT_MODEL = "speechbrain/sepformer-wsj02mix"


def _token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


@register_separator
class HuggingFaceAPISeparator(BaseSeparator):
    info = MethodInfo(
        key="api_huggingface",
        name="Hugging Face audio endpoint",
        kind="separate",
        family="api",
        description=(
            "Runs a separator through a configured Hugging Face audio endpoint. "
            "Requires hosted inference, not just a model uploaded to the Hub."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=False,
        pip=["requests"],
        install_hint="pip install requests, then configure HF_TOKEN and HF_SEPARATION_URL or HF_SEPARATION_MODEL",
        notes="UPLOADS YOUR AUDIO. Requires a token and a deployed compatible audio endpoint; costs may apply.",
        max_speakers=3,
    )
    target_sr = 8000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("requests"):
            return False, "missing python package(s): requests"
        if not _token():
            return False, "set HF_TOKEN (free huggingface.co read token) to enable this method"
        if not os.environ.get('HF_SEPARATION_URL', '').strip() and not os.environ.get('HF_SEPARATION_MODEL', '').strip():
            return False, 'configure HF_SEPARATION_URL or a hosted HF_SEPARATION_MODEL; a token alone does not deploy the model'
        return True, ""

    def load(self) -> None:
        self._model_id = os.environ.get("HF_SEPARATION_MODEL", DEFAULT_MODEL)
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        model_id = self._model_id
        if num_speakers and num_speakers >= 3 and "wsj02mix" in model_id:
            model_id = model_id.replace("wsj02mix", "wsj03mix")

        from .sepformer import MODEL_TABLE
        model_sr, _ = MODEL_TABLE.get(model_id, (self.target_sr, 2))
        work = resample(audio, model_sr)
        resp, attempts = request_audio(work, model_id, _token(), 'separation', timeout=300)
        sources, labels, out_sr = self._parse(resp)
        if not sources:
            raise RuntimeError("API returned no audio sources")
        return sources, {
            "backend": "hf-api", "output_sr": out_sr, "confidence": 0.7, "labels": labels,
            "metrics": {"model_id": model_id, "attempts": attempts, "off_device": True},
        }

    @staticmethod
    def _parse(resp) -> Tuple[List[np.ndarray], List[str], int]:
        sources: List[np.ndarray] = []
        labels: List[str] = []
        out_sr = 8000
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, dict):
                data = [data]
            for i, item in enumerate(data or []):
                blob = item.get("blob") if isinstance(item, dict) else None
                if not blob:
                    continue
                buf = load_from_bytes(base64.b64decode(blob), "src%d.wav" % i)
                if not sources:
                    out_sr = buf.sr
                elif buf.sr != out_sr:
                    buf = resample(buf, out_sr)
                sources.append(buf.samples)
                labels.append(str(item.get("label") or ("Speaker %d" % (i + 1))))
        else:
            buf = load_from_bytes(resp.content, "src.wav")
            out_sr = buf.sr
            sources.append(buf.samples)
            labels.append("Speaker 1")
        return sources, labels, out_sr
