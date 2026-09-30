"""Human-readable errors.

**Rule: a Python traceback never reaches the browser.**

The UI gets a short sentence saying what went wrong plus, wherever we can work
it out, a concrete fix ("install ffmpeg", "pick a shorter clip"). The full
traceback still exists -- it goes to the server console and to
``data/cache/errors.log`` -- so debugging is unaffected.

Usage::

    from ..core.errors import AppError, describe_exception, log_exception

    try:
        ...
    except Exception as exc:
        log_exception("denoise", exc)              # full detail, server-side
        info = describe_exception(exc)             # {"message": ..., "fix": ...}
        emit(info["message"])
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, Optional

MAX_LOG_BYTES = 2 * 1024 * 1024


class AppError(Exception):
    """An error we raised on purpose, already phrased for a human.

    ``message`` says what went wrong, ``fix`` says what to do about it.
    """

    def __init__(self, message: str, fix: str = "", kind: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.kind = kind

    def to_dict(self) -> Dict[str, str]:
        return {"message": self.message, "fix": self.fix, "kind": self.kind}


class MissingDependency(AppError):
    """A backend needs a package / binary that is not installed."""

    def __init__(self, what: str, install: str) -> None:
        super().__init__("%s is not installed." % what, fix=install, kind="missing-dependency")


class UnreadableAudio(AppError):
    """The file could not be decoded."""


# --------------------------------------------------------------------------- #
#  turning an arbitrary exception into something a human can act on
# --------------------------------------------------------------------------- #
def describe_exception(exc: BaseException, context: str = "") -> Dict[str, str]:
    """Map any exception to ``{"message", "fix", "kind"}`` -- never a traceback."""
    if isinstance(exc, AppError):
        data = exc.to_dict()
        if context and not data["message"].lower().startswith(context.lower()):
            data["message"] = "%s: %s" % (context.capitalize(), data["message"])
        return data

    name = type(exc).__name__
    text = _first_line(str(exc))

    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        missing = getattr(exc, "name", None) or text
        return _pack(
            "A required package (%s) is not installed." % missing,
            "Install it, then reload this page. See the method list for the exact command.",
            "missing-dependency", context,
        )
    if isinstance(exc, FileNotFoundError):
        return _pack("The file could not be found any more.",
                     "It may have been moved or deleted -- upload it again.",
                     "missing-file", context)
    if isinstance(exc, PermissionError):
        return _pack("The file could not be opened (permission denied).",
                     "Close the file in other programs, or copy it somewhere writable.",
                     "permission", context)
    if isinstance(exc, MemoryError):
        return _pack("Ran out of memory while processing this file.",
                     "Try a shorter clip, or a lighter method (spectral_gate / diarize_cluster).",
                     "memory", context)
    if isinstance(exc, TimeoutError) or "timed out" in text.lower():
        return _pack("The operation timed out.",
                     "This usually means a very long file or a slow network method.",
                     "timeout", context)
    if name in {"ConnectionError", "HTTPError", "SSLError", "ReadTimeout", "ConnectTimeout"} \
            or "connection" in text.lower():
        return _pack("Could not reach the online service.",
                     "Check your internet connection, or pick a local (offline) method.",
                     "network", context)
    if "cuda" in text.lower() and "out of memory" in text.lower():
        return _pack("The GPU ran out of memory.",
                     "Use a shorter clip, or set CUDA_VISIBLE_DEVICES= to force CPU.",
                     "gpu-memory", context)

    # Unknown: keep the exception's own sentence (it is usually the useful part)
    # but never the traceback, file paths or line numbers.
    message = text or name
    if len(message) > 300:
        message = message[:297] + "..."
    return _pack(message, "", "error", context)


def _pack(message: str, fix: str, kind: str, context: str) -> Dict[str, str]:
    if context:
        message = "%s: %s" % (context.capitalize(), message)
    return {"message": message, "fix": fix, "kind": kind}


def _first_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def error_summary(exc: BaseException, context: str = "") -> str:
    """One string, message + fix, for logs and single-line UI slots."""
    info = describe_exception(exc, context)
    return "%s %s" % (info["message"], info["fix"]) if info["fix"] else info["message"]


# --------------------------------------------------------------------------- #
#  server-side logging (where the traceback DOES go)
# --------------------------------------------------------------------------- #
def log_exception(where: str, exc: BaseException, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write the full traceback to stderr and to data/cache/errors.log."""
    import time

    header = "[%s] %s -- %s: %s" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), where, type(exc).__name__, _first_line(str(exc))
    )
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    blob = header + "\n" + (("  context: %r\n" % extra) if extra else "") + body + "\n"

    print(header, file=sys.stderr, flush=True)
    try:
        from .utils import CACHE_DIR

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / "errors.log"
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            path.write_text("", encoding="utf-8")  # simple truncation, no rotation needed
        with path.open("a", encoding="utf-8") as fh:
            fh.write(blob)
    except Exception:
        # logging must never be the thing that breaks a run
        print(body, file=sys.stderr, flush=True)
