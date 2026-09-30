"""Importing this package registers every denoising backend.

Each import is guarded: a backend whose *module* fails to import (bad install,
python version mismatch, ...) is skipped and recorded in :data:`IMPORT_ERRORS`
instead of taking the whole app down.
"""

from __future__ import annotations

import importlib
from typing import Dict, List

MODULES: List[str] = [
    "passthrough",
    "spectral_gate",
    "wiener_mmse",
    "rnnoise",
    "deepfilternet",
    "demucs_denoiser",
    "demucs_vocals",
    "resemble_enhance",
    "local_unet",
    "api_huggingface",
]

IMPORT_ERRORS: Dict[str, str] = {}

for _name in MODULES:
    try:
        importlib.import_module("." + _name, __name__)
    except Exception as _exc:  # pragma: no cover - defensive
        IMPORT_ERRORS[_name] = "%s: %s" % (type(_exc).__name__, _exc)

__all__ = ["MODULES", "IMPORT_ERRORS"]
