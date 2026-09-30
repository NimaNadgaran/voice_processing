"""Small helpers: project paths, timing, ids, json-safe conversion, dep probing."""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

# --------------------------------------------------------------------------- #
#  project paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"
DATASET_DIR = DATA_DIR / "datasets"
MODELS_DIR = ROOT / "models"
CKPT_DIR = MODELS_DIR / "checkpoints"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
FRONTEND_DIR = ROOT / "frontend"
CONFIG_DIR = ROOT / "config"


def ensure_dirs() -> None:
    for d in (RAW_DIR, OUT_DIR, CACHE_DIR, DATASET_DIR, CKPT_DIR, PRETRAINED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  ids / names / formatting
# --------------------------------------------------------------------------- #
def new_id(prefix: str = "job") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return prefix + "_" + stamp + "_" + uuid.uuid4().hex[:6]


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str, fallback: str = "audio") -> str:
    stem = Path(str(name)).name
    stem = _SAFE.sub("_", stem).strip("._-")
    return stem or fallback


def validate_component(value: str) -> str:
    """Validate an output directory component on both Windows and POSIX."""
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value)
            or ".." in value or value.endswith(".")
            or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *["COM%d" % i for i in range(1, 10)], *["LPT%d" % i for i in range(1, 10)]}):
        raise ValueError("invalid output id: use letters, digits, underscores and hyphens")
    return value


def contained_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("path traversal rejected")
    return target


def human_time(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 1:
        return "%.0f ms" % (seconds * 1000)
    if seconds < 60:
        return "%.2f s" % seconds
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return "%dm %.0fs" % (int(minutes), sec)
    hours, minutes = divmod(minutes, 60)
    return "%dh %dm" % (int(hours), int(minutes))


def human_size(num_bytes: float) -> str:
    num = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return ("%d B" % int(num)) if unit == "B" else ("%.1f %s" % (num, unit))
        num /= 1024.0
    return "%.1f GB" % num


# --------------------------------------------------------------------------- #
#  timing
# --------------------------------------------------------------------------- #
class Timer:
    """``with Timer() as t: ...``  ->  ``t.elapsed``"""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def timed(store: Dict[str, Any], key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0


# --------------------------------------------------------------------------- #
#  json safety (numpy scalars are not json serialisable)
# --------------------------------------------------------------------------- #
def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else round(val, 6)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    return str(obj)


def write_json(path: "os.PathLike[str] | str", payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: "os.PathLike[str] | str", default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


# --------------------------------------------------------------------------- #
#  dependency probing (never raises, never imports the heavy module)
# --------------------------------------------------------------------------- #
def module_available(name: str) -> bool:
    import importlib.util

    try:
        top = name.split(".")[0]
        if importlib.util.find_spec(top) is None:
            return False
        if "." in name:
            return importlib.util.find_spec(name) is not None
        return True
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        return False


def missing_modules(names: Iterable[str]) -> List[str]:
    return [n for n in names if not module_available(n)]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
