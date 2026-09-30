"""Method 9 -- configured Hugging Face speech-enhancement endpoint.

Setup
-----
1. Make a free account at huggingface.co and create a read token.
2. ``set HF_TOKEN=hf_xxx``  (Windows)  /  ``export HF_TOKEN=hf_xxx``  (Unix)
3. Set HF_DENOISE_URL to a deployed compatible audio endpoint, or set
   HF_DENOISE_MODEL to a model actually hosted by the inference provider.

PRIVACY NOTE -- this is the only denoiser that leaves your machine.  Your audio
is uploaded to Hugging Face's servers for inference.  It is therefore *opt-in*:
without ``HF_TOKEN`` the method stays disabled, and the UI shows a "sends audio
off-device" badge next to it.  Every other method in this project is 100%
local.

Being present on the Hub does not mean a model has hosted inference. Endpoints
must accept WAV uploads and return WAV bytes or JSON audio blobs; costs and
access permissions depend on the configured service.
"""

from __future__ import annotations

import base64
import os
from typing import Tuple

from ...core.audio_io import load_from_bytes
from ...core.hosted_audio import API_ROOT, request_audio
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser

DEFAULT_MODEL = "speechbrain/sepformer-wham16k-enhancement"


def _token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


@register_denoiser
class HuggingFaceAPIDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="api_huggingface",
        name="Hugging Face audio endpoint",
        kind="denoise",
        family="api",
        description=(
            "Runs a speech-enhancement model through a configured Hugging Face "
            "audio endpoint. Requires hosted inference, not just a Hub model."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=False,
        pip=["requests"],
        install_hint="pip install requests, then configure HF_TOKEN and HF_DENOISE_URL or HF_DENOISE_MODEL",
        notes="UPLOADS YOUR AUDIO. Requires a token and a deployed compatible audio endpoint; costs may apply.",
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("requests"):
            return False, "missing python package(s): requests"
        if not _token():
            return False, "set HF_TOKEN (free huggingface.co read token) to enable this method"
        if not os.environ.get('HF_DENOISE_URL', '').strip() and not os.environ.get('HF_DENOISE_MODEL', '').strip():
            return False, 'configure HF_DENOISE_URL or a hosted HF_DENOISE_MODEL; a token alone does not deploy the model'
        return True, ""

    def load(self) -> None:
        self._model_id = os.environ.get("HF_DENOISE_MODEL", DEFAULT_MODEL)
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        resp, attempts = request_audio(audio, self._model_id, _token(), 'denoise')
        out, info = self._parse(resp, audio)
        info.update(attempts=attempts, off_device=True, model_id=self._model_id)
        return out, info

    @staticmethod
    def _parse(resp, original: AudioBuffer):
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list) and data and "blob" in data[0]:
                raw = base64.b64decode(data[0]["blob"])
                out = load_from_bytes(raw, "hf.wav")
                return out, {"backend": "hf-api", "label": data[0].get("label", "")}
            raise RuntimeError("unexpected JSON from the API: %s" % str(data)[:200])
        out = load_from_bytes(resp.content, "hf.wav")
        return out, {"backend": "hf-api", "bytes": len(resp.content)}
