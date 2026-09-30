"""Method 8 -- Hugging Face Inference API (free tier) source separation.

Setup
-----
1. Free account on huggingface.co -> create a read token.
2. ``set HF_TOKEN=hf_xxx`` (Windows) / ``export HF_TOKEN=hf_xxx`` (Unix).

PRIVACY NOTE -- this is the only separator that leaves your machine: the audio
is uploaded to Hugging Face for inference.  It is therefore opt-in (no token =
method disabled) and the UI marks it with an "off-device" badge.

The audio-to-audio pipeline returns one base64 blob per separated source, which
we decode straight into speaker tracks.  Free-tier endpoints are rate limited
and cold-start slowly (503 while the model loads -- we retry).

Change the model with ``HF_SEPARATION_MODEL``; any audio-to-audio model works,
e.g. ``speechbrain/sepformer-wsj02mix`` (2 spk) or
``speechbrain/sepformer-wsj03mix`` (3 spk).
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from ...core.audio_io import load_from_bytes, save_audio
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CACHE_DIR, module_available, new_id
from ..base import BaseSeparator

DEFAULT_MODEL = "speechbrain/sepformer-wsj02mix"
API_ROOT = "https://api-inference.huggingface.co/models/"


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
        name="Hugging Face API (free tier)",
        kind="separate",
        family="api",
        description=(
            "Runs a hosted SepFormer through the free Hugging Face Inference API. "
            "Zero local compute and no multi-GB installs -- handy on weak machines."
        ),
        speed="medium",
        quality=4,
        needs_gpu=False,
        offline=False,
        pip=["requests"],
        install_hint="pip install requests, then set HF_TOKEN=hf_xxx (free account)",
        notes="UPLOADS YOUR AUDIO to Hugging Face. Disabled unless HF_TOKEN is set. Rate limited.",
        max_speakers=3,
    )
    target_sr = 8000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("requests"):
            return False, "missing python package(s): requests"
        if not _token():
            return False, "set HF_TOKEN (free huggingface.co read token) to enable this method"
        return True, ""

    def load(self) -> None:
        self._model_id = os.environ.get("HF_SEPARATION_MODEL", DEFAULT_MODEL)
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        import requests  # type: ignore

        model_id = self._model_id
        if num_speakers and num_speakers >= 3 and "wsj02mix" in model_id:
            model_id = model_id.replace("wsj02mix", "wsj03mix")

        tmp = CACHE_DIR / (new_id("hfsep") + ".wav")
        save_audio(tmp, audio)
        try:
            payload = tmp.read_bytes()
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

        headers = {"Authorization": "Bearer " + _token(), "Content-Type": "audio/wav"}
        url = API_ROOT + model_id

        last = ""
        for attempt in range(4):
            resp = requests.post(url, headers=headers, data=payload, timeout=300)
            if resp.status_code == 200:
                sources, labels, out_sr = self._parse(resp)
                if not sources:
                    raise RuntimeError("API returned no audio sources")
                return sources, {
                    "backend": "hf-api",
                    "output_sr": out_sr,
                    "confidence": 0.7,
                    "labels": labels,
                    "metrics": {"model_id": model_id, "attempts": attempt + 1, "off_device": True},
                }
            last = "HTTP %d: %s" % (resp.status_code, resp.text[:300])
            if resp.status_code == 503:
                wait = 8.0 * (attempt + 1)
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
                out_sr = buf.sr
                sources.append(buf.samples)
                labels.append(str(item.get("label") or ("Speaker %d" % (i + 1))))
        else:
            buf = load_from_bytes(resp.content, "src.wav")
            out_sr = buf.sr
            sources.append(buf.samples)
            labels.append("Speaker 1")
        return sources, labels, out_sr
