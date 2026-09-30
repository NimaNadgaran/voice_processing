"""Plugin registry.

Every backend is a class decorated with :func:`register_denoiser` /
:func:`register_separator`.  Adding a new method is literally:

    from src.core.registry import register_denoiser
    from src.denoising.base import BaseDenoiser

    @register_denoiser
    class MyDenoiser(BaseDenoiser):
        info = MethodInfo(key="mine", ...)
        def _denoise(self, audio): ...

The registry never imports heavy dependencies at declaration time -- each class
answers :meth:`is_available` by *probing* (importlib.util.find_spec) so the web
UI can list a method as "not installed" instead of crashing.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Type

from .types import MethodInfo

DENOISERS: Dict[str, Type[Any]] = {}
SEPARATORS: Dict[str, Type[Any]] = {}
TRANSCRIBERS: Dict[str, Type[Any]] = {}

_INSTANCES: Dict[str, Any] = {}
_LOCK = threading.Lock()
_LOADED = False


# --------------------------------------------------------------------------- #
#  decorators
# --------------------------------------------------------------------------- #
def register_denoiser(cls: Type[Any]) -> Type[Any]:
    key = cls.info.key
    if key in DENOISERS:
        raise ValueError("duplicate denoiser key: %s" % key)
    DENOISERS[key] = cls
    return cls


def register_separator(cls: Type[Any]) -> Type[Any]:
    key = cls.info.key
    if key in SEPARATORS:
        raise ValueError("duplicate separator key: %s" % key)
    SEPARATORS[key] = cls
    return cls


def register_transcriber(cls: Type[Any]) -> Type[Any]:
    key = cls.info.key
    if key in TRANSCRIBERS:
        raise ValueError('duplicate transcriber key: ' + key)
    TRANSCRIBERS[key] = cls
    return cls


# --------------------------------------------------------------------------- #
#  discovery
# --------------------------------------------------------------------------- #
def load_all() -> None:
    """Import the method packages once so the decorators run."""
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        import src.denoising.methods  # noqa: F401  (side-effect import)
        import src.separation.methods  # noqa: F401
        import src.transcription.methods  # noqa: F401

        _LOADED = True


# --------------------------------------------------------------------------- #
#  accessors
# --------------------------------------------------------------------------- #
def _get(table: Dict[str, Type[Any]], key: str, kind: str) -> Any:
    load_all()
    if key not in table:
        raise KeyError(
            "unknown %s '%s'. available: %s" % (kind, key, ", ".join(sorted(table)))
        )
    cache_key = "%s:%s" % (kind, key)
    with _LOCK:
        inst = _INSTANCES.get(cache_key)
        if inst is None:
            inst = table[key]()
            _INSTANCES[cache_key] = inst
    return inst


def get_denoiser(key: str) -> Any:
    return _get(DENOISERS, key, "denoiser")


def get_separator(key: str) -> Any:
    return _get(SEPARATORS, key, "separator")


def get_transcriber(key: str) -> Any:
    return _get(TRANSCRIBERS, key, 'transcriber')


def _describe(table: Dict[str, Type[Any]], only_available: bool) -> List[MethodInfo]:
    load_all()
    out: List[MethodInfo] = []
    for key in table:
        try:
            kind = 'denoiser' if table is DENOISERS else 'separator' if table is SEPARATORS else 'transcriber'
            inst = _get(table, key, kind)
            info = inst.describe()
        except Exception as exc:  # a broken plugin must never hide the others
            info = MethodInfo(
                key=key,
                name=key,
                kind="denoise" if table is DENOISERS else "separate" if table is SEPARATORS else 'transcribe',
                family="unknown",
                description="failed to initialise",
                available=False,
                unavailable_reason=str(exc),
            )
        if only_available and not info.available:
            continue
        out.append(info)
    order = {"dsp": 0, "deep-pretrained": 1, "deep-local": 2, "api": 3}
    out.sort(key=lambda m: (order.get(m.family, 9), not m.available, m.key))
    return out


def list_denoisers(only_available: bool = False) -> List[MethodInfo]:
    return _describe(DENOISERS, only_available)


def list_separators(only_available: bool = False) -> List[MethodInfo]:
    return _describe(SEPARATORS, only_available)


def list_transcribers(only_available: bool = False) -> List[MethodInfo]:
    return _describe(TRANSCRIBERS, only_available)


def first_available(keys: List[str], kind: str = "denoise") -> Optional[str]:
    """Return the first key in ``keys`` whose backend is installed."""
    table = DENOISERS if kind == "denoise" else SEPARATORS
    load_all()
    for key in keys:
        if key not in table:
            continue
        try:
            if _get(table, key, kind).describe().available:
                return key
        except Exception:
            continue
    return None


def clear_instance_cache() -> None:
    """Drop loaded models (frees RAM / VRAM)."""
    with _LOCK:
        _INSTANCES.clear()
