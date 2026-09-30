"""Base class every denoiser inherits from.

Contract for a subclass
-----------------------
* declare a class-level :class:`MethodInfo` called ``info``
* optionally set ``target_sr`` (the rate the backend wants -- the base class
  resamples in and back out for you)
* implement ``_denoise(self, audio: AudioBuffer) -> AudioBuffer`` (or return a
  ``(AudioBuffer, dict)`` tuple to attach extra metrics)
* optionally override ``check_available()`` and ``load()``

Everything else -- timing, resampling, length matching, NaN guarding, metric
computation, error handling -- is handled here so all backends behave the same.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..core.audio_io import match_length, resample
from ..core.errors import AppError, describe_exception, log_exception
from ..core.metrics import denoise_metrics
from ..core.types import AudioBuffer, DenoiseResult, MethodInfo
from ..core.utils import Timer, missing_modules
from .explanations import DETAILS

ProgressFn = Optional[Callable[[float, str], None]]


class DenoiserUnavailable(AppError):
    """Raised when a backend's dependencies or weights are missing.

    Carries the install command so the UI can show an actionable fix.
    """


class BaseDenoiser:
    info: MethodInfo = MethodInfo(
        key="base", name="Base", kind="denoise", family="dsp", description=""
    )
    target_sr: Optional[int] = None  # None = process at the input rate
    restore_sr: bool = True  # resample the output back to the input rate

    def __init__(self) -> None:
        self._loaded = False
        self._model: Any = None
        self._load_error: str = ""

    # ------------------------------------------------------------------ #
    #  availability
    # ------------------------------------------------------------------ #
    def check_available(self) -> Tuple[bool, str]:
        """Override for extra checks (weights on disk, API key present, ...)."""
        missing = missing_modules(self.info.pip)
        if missing:
            return False, "missing python package(s): " + ", ".join(missing)
        return True, ""

    def describe(self) -> MethodInfo:
        info = copy.deepcopy(self.info)
        try:
            ok, reason = self.check_available()
        except Exception as exc:
            ok, reason = False, "probe failed: %s" % exc
        info.available = bool(ok)
        info.unavailable_reason = "" if ok else reason

        # merge the long-form explanation shown in the UI's "How this works"
        detail = DETAILS.get(info.key, {})
        for field_name in ("how_it_works", "steps", "strengths", "limitations",
                           "reference", "latency"):
            value = detail.get(field_name)
            if value:
                setattr(info, field_name, value)
        return info

    # ------------------------------------------------------------------ #
    #  lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Heavy one-time setup (model download / weight loading)."""
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ok, reason = self.check_available()
        if not ok:
            raise DenoiserUnavailable(
                "%s is not ready: %s" % (self.info.name, reason),
                fix=self.info.install_hint,
                kind="missing-dependency",
            )
        self.load()
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  the actual work (subclass implements this)
    # ------------------------------------------------------------------ #
    def _denoise(self, audio: AudioBuffer) -> "AudioBuffer | Tuple[AudioBuffer, Dict[str, Any]]":
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  public entry point
    # ------------------------------------------------------------------ #
    def run(self, audio: AudioBuffer, progress: ProgressFn = None) -> DenoiseResult:
        logs: list = []

        def emit(pct: float, msg: str) -> None:
            logs.append(msg)
            if progress:
                progress(pct, msg)

        emit(0.02, "preparing %s" % self.info.name)
        self._ensure_loaded()

        source_sr = audio.sr
        work = audio
        if self.target_sr and audio.sr != self.target_sr:
            emit(0.08, "resampling %d Hz -> %d Hz" % (audio.sr, self.target_sr))
            work = resample(audio, self.target_sr)

        emit(0.15, "running %s" % self.info.name)
        extra: Dict[str, Any] = {}
        with Timer() as timer:
            out = self._denoise(work)
            if isinstance(out, tuple):
                out, extra = out
        emit(0.80, "enhancement finished in %.2fs" % timer.elapsed)

        if not isinstance(out, AudioBuffer):
            out = AudioBuffer(np.asarray(out, dtype=np.float32), work.sr)

        out.samples = np.nan_to_num(out.samples, nan=0.0, posinf=0.0, neginf=0.0)

        if self.restore_sr and out.sr != source_sr:
            emit(0.86, "restoring original sample rate (%d Hz)" % source_sr)
            out = resample(out, source_sr)
        if out.sr == source_sr:
            out = AudioBuffer(match_length(out.samples, audio.n_samples), out.sr)

        emit(0.92, "measuring quality")
        metrics = denoise_metrics(audio, out)
        metrics.update(extra)

        emit(1.0, "done (%s)" % self.info.name)
        return DenoiseResult(
            audio=out,
            method=self.info.key,
            elapsed=timer.elapsed,
            metrics=metrics,
            logs=logs,
            backend_used=extra.get("backend", self.info.key),
        )

    # ------------------------------------------------------------------ #
    def safe_run(self, audio: AudioBuffer, progress: ProgressFn = None) -> DenoiseResult:
        """Never raises -- returns the input untouched with the error attached."""
        try:
            return self.run(audio, progress)
        except Exception as exc:
            log_exception("denoiser:%s" % self.info.key, exc)
            info = describe_exception(exc)
            return DenoiseResult(
                audio=audio.copy(),
                method=self.info.key,
                elapsed=0.0,
                metrics={"error": info["message"], "error_fix": info["fix"]},
                logs=[info["message"]] + ([info["fix"]] if info["fix"] else []),
                backend_used="failed",
            )
