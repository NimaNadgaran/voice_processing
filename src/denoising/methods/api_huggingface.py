"""Method 9 -- Hugging Face Inference API (free tier) speech enhancement.

Setup
-----
1. Make a free account at huggingface.co and create a read token.
2. ``set HF_TOKEN=hf_xxx``  (Windows)  /  ``export HF_TOKEN=hf_xxx``  (Unix)
3. The method turns itself on in the UI.

PRIVACY NOTE -- this is the only denoiser that leaves your machine.  Your audio
is uploaded to Hugging Face's servers for inference.  It is therefore *opt-in*:
without ``HF_TOKEN`` the method stays disabled, and the UI shows a "sends audio
off-device" badge next to it.  Every other method in this project is 100%
local.

Free alternatives with the same shape (drop the base URL / model id into the
env vars below):
* ``HF_DENOISE_MODEL``   -- any audio-to-audio model on the Hub
* self-hosted `text-generation-inference`-style endpoints
* a local ``docker run`` of the same model (then it is offline again)
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Tuple

from ...core.audio_io import load_from_bytes, save_audio
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CACHE_DIR, module_available
from ..base import BaseDenoiser

DEFAULT_MODEL = "speechbrain/sepformer-wham16k-enhancement"
API_ROOT = "https://api-inference.huggingface.co/models/"


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
        name="Hugging Face API (free tier)",
        kind="denoise",
        family="api",
        description=(
            "Runs a hosted speech-enhancement model through the free Hugging Face "
            "Inference API. Zero local compute -- useful on very weak machines."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=False,
        pip=["requests"],
        install_hint="pip install requests, then set HF_TOKEN=hf_xxx (free account)",
        notes="UPLOADS YOUR AUDIO to Hugging Face. Disabled unless HF_TOKEN is set. Free tier is rate limited.",
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("requests"):
            return False, "missing python package(s): requests"
        if not _token():
            return False, "set HF_TOKEN (free huggingface.co read token) to enable this method"
        return True, ""

    def load(self) -> None:
        self._model_id = os.environ.get("HF_DENOISE_MODEL", DEFAULT_MODEL)
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        import requests  # type: ignore

        tmp = CACHE_DIR / "hf_upload.wav"
        save_audio(tmp, audio)
        payload = tmp.read_bytes()

        url = API_ROOT + self._model_id
        headers = {"Authorization": "Bearer " + _token(), "Content-Type": "audio/wav"}

        last = ""
        for attempt in range(4):
            resp = requests.post(url, headers=headers, data=payload, timeout=180)
            if resp.status_code == 200:
                return self._parse(resp, audio)
            last = "HTTP %d: %s" % (resp.status_code, resp.text[:300])
            if resp.status_code == 503:  # cold start -- the model is loading
                wait = 8 * (attempt + 1)
                try:
                    wait = float(json.loads(resp.text).get("estimated_time", wait))
                except Exception:
                    pass
                time.sleep(min(wait, 40))
                continue
            if resp.status_code in (401, 403):
                raise RuntimeError("Hugging Face rejected the token (%s)" % last)
            if resp.status_code == 429:
                time.sleep(10)
                continue
            break
        raise RuntimeError("Hugging Face API call failed -- " + last)

    @staticmethod
    def _parse(resp, original: AudioBuffer):
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, list) and data and "blob" in data[0]:
                raw = base64.b64decode(data[0]["blob"])
                out = load_from_bytes(raw, "hf.wav")
                return out, {"backend": "hf-api", "label": data[0].get("label", "")}
            raise RuntimeError("unexpected JSON from the API: %s" % str(data)[:200])
        out = load_from_bytes(resp.content, "hf.wav")
        return out, {"backend": "hf-api", "bytes": len(resp.content)}
