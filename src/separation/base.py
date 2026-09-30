"""Base class every speaker separator inherits from.

Contract for a subclass
-----------------------
* class-level :class:`MethodInfo` named ``info``
* optional ``target_sr``
* implement ``_separate(self, audio, num_speakers) -> List[np.ndarray]``
  (or return ``(sources, extra_metrics_dict)``)

The base class then does all the shared bookkeeping:

* resample in / out,
* drop sources that are silent or are duplicates of a louder source
  (a 2-speaker model fed a 1-speaker file emits a near-empty second stem),
* per-track VAD -> speech segments, talk time, energy,
* a per-track confidence and an overall reference-free separation score,
* timing and error capture.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.audio_io import match_length, resample
from ..core.errors import AppError, describe_exception, log_exception
from ..core.dsp import energy_vad, mask_to_segments
from ..core.metrics import estimated_snr, separation_metrics, signal_stats
from ..core.types import AudioBuffer, MethodInfo, SeparationResult, SpeakerTrack
from ..core.utils import Timer, missing_modules
from .explanations import DETAILS

ProgressFn = Optional[Callable[[float, str], None]]


class SeparatorUnavailable(AppError):
    """Raised when a backend's dependencies, weights or token are missing.

    Carries the install command so the UI can show an actionable fix.
    """


class BaseSeparator:
    info: MethodInfo = MethodInfo(
        key="base", name="Base", kind="separate", family="dsp", description=""
    )
    target_sr: Optional[int] = None
    restore_sr: bool = True

    #: sources quieter than (loudest - this) dB are considered empty stems
    silence_floor_db: float = 32.0
    #: minimum seconds of detected speech for a track to be kept
    min_speech_seconds: float = 0.35

    def __init__(self) -> None:
        self._run_lock = threading.RLock()
        self._loaded = False
        self._model: Any = None

    # ------------------------------------------------------------------ #
    #  availability
    # ------------------------------------------------------------------ #
    def check_available(self) -> Tuple[bool, str]:
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
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ok, reason = self.check_available()
        if not ok:
            raise SeparatorUnavailable(
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
    #  subclass implements this
    # ------------------------------------------------------------------ #
    def _separate(
        self, audio: AudioBuffer, num_speakers: Optional[int]
    ) -> "List[np.ndarray] | Tuple[List[np.ndarray], Dict[str, Any]]":
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  public entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        audio: AudioBuffer,
        num_speakers: Optional[int] = None,
        progress: ProgressFn = None,
    ) -> SeparationResult:
        with self._run_lock:
            return self._run(audio, num_speakers, progress)

    def _run(self, audio, num_speakers=None, progress=None):
        logs: List[str] = []

        def emit(pct: float, msg: str) -> None:
            logs.append(msg)
            if progress:
                progress(pct, msg)

        emit(0.02, "preparing %s" % self.info.name)
        self._ensure_loaded()
        if audio.n_samples == 0:
            return SeparationResult([], self.info.key, 0., 0, 0., metrics={"empty_input": True})

        source_sr = audio.sr
        work = audio
        if self.target_sr and audio.sr != self.target_sr:
            emit(0.08, "resampling %d Hz -> %d Hz" % (audio.sr, self.target_sr))
            work = resample(audio, self.target_sr)

        cap = self.info.max_speakers
        if num_speakers and cap and num_speakers > cap:
            emit(0.1, "note: %s handles at most %d speakers (asked for %d)"
                 % (self.info.name, cap, num_speakers))

        emit(0.15, "separating speakers")
        extra: Dict[str, Any] = {}
        with Timer() as timer:
            out = self._separate(work, num_speakers)
            if isinstance(out, tuple):
                out, extra = out
        emit(0.7, "model finished in %.2fs -- %d raw stems" % (timer.elapsed, len(out)))

        # a backend may resample internally and tell us via extra["output_sr"]
        out_sr = int(extra.get("output_sr", work.sr))
        buffers, indices = self._postprocess(out, out_sr, source_sr, audio.n_samples,
                                             return_indices=True, segments=extra.get("segments"))
        emit(0.8, "kept %d speaker track(s) after cleanup" % len(buffers))

        labels = extra.get("labels")
        segments = extra.get("segments")
        tracks = self._build_tracks(buffers,
                                    [labels[i] for i in indices] if labels else None,
                                    [segments[i] for i in indices] if segments else None)
        emit(0.9, "measuring separation quality")

        metrics = separation_metrics(audio, [t.audio for t in tracks])
        metrics.update(extra.get("metrics", {}))
        confidence = float(extra.get("confidence", metrics.get("separation_score", 0.0) / 100.0))
        for t in tracks:
            t.confidence = float(np.clip(confidence * (0.6 + 0.4 * min(t.speech_ratio * 4, 1.0)), 0, 1))

        emit(1.0, "done -- %d speaker(s)" % len(tracks))
        return SeparationResult(
            tracks=tracks,
            method=self.info.key,
            elapsed=timer.elapsed,
            n_speakers=len(tracks),
            n_speakers_confidence=confidence,
            metrics=metrics,
            logs=logs,
            backend_used=extra.get("backend", self.info.key),
        )

    # ------------------------------------------------------------------ #
    def safe_run(
        self,
        audio: AudioBuffer,
        num_speakers: Optional[int] = None,
        progress: ProgressFn = None,
    ) -> SeparationResult:
        try:
            return self.run(audio, num_speakers, progress)
        except Exception as exc:
            log_exception("separator:%s" % self.info.key, exc)
            info = describe_exception(exc)
            return SeparationResult(
                tracks=[],
                method=self.info.key,
                elapsed=0.0,
                n_speakers=0,
                metrics={"error": info["message"], "error_fix": info["fix"]},
                logs=[info["message"]] + ([info["fix"]] if info["fix"] else []),
                backend_used="failed",
            )

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #
    def _postprocess(
        self,
        sources: Sequence[np.ndarray],
        out_sr: int,
        source_sr: int,
        n_target: int,
        return_indices=False,
        segments=None,
    ) -> List[AudioBuffer]:
        bufs: List[AudioBuffer] = []
        for src in sources:
            arr = np.nan_to_num(np.asarray(src, dtype=np.float32).reshape(-1),
                                nan=0., posinf=0., neginf=0.)
            buf = AudioBuffer(arr, out_sr)
            if self.restore_sr and buf.sr != source_sr:
                buf = resample(buf, source_sr)
            if buf.sr == source_sr:
                buf = AudioBuffer(match_length(buf.samples, n_target), source_sr)
            bufs.append(buf)

        if not bufs:
            return (bufs, []) if return_indices else bufs

        energies = np.array([float(np.sqrt(np.mean(b.samples**2)) + 1e-12) for b in bufs])
        loudest = float(energies.max())
        keep: List[AudioBuffer] = []
        indices = []
        for i, (buf, energy) in enumerate(zip(bufs, energies)):
            # Neural diarizers already know which turns are speech. Re-running
            # the energy VAD can erase short/quiet speakers and corrupt labels.
            known_speech = bool(segments and i < len(segments) and segments[i])
            rel_db = 20 * np.log10(energy / loudest)
            if rel_db < -self.silence_floor_db and not known_speech:
                continue
            mask, hop = energy_vad(buf.samples, buf.sr)
            if float(mask.sum() * hop) < min(self.min_speech_seconds, buf.duration / 2) and not known_speech:
                continue
            keep.append(buf)
            indices.append(i)

        # never return nothing: fall back to the loudest raw stem
        if not keep:
            keep = [bufs[int(np.argmax(energies))]]
            indices = [int(np.argmax(energies))]
        return (keep, indices) if return_indices else keep

    @staticmethod
    def _build_tracks(
        buffers: Sequence[AudioBuffer],
        labels: Optional[Sequence[str]] = None,
        segments: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ) -> List[SpeakerTrack]:
        tracks: List[SpeakerTrack] = []
        for i, buf in enumerate(buffers):
            mask, hop = energy_vad(buf.samples, buf.sr)
            segs = list(segments[i]) if segments and i < len(segments) else mask_to_segments(mask, hop)
            snr = estimated_snr(buf)
            stats = signal_stats(buf)
            tracks.append(
                SpeakerTrack(
                    index=i,
                    audio=buf,
                    label=(labels[i] if labels and i < len(labels) else "Speaker %d" % (i + 1)),
                    speech_ratio=float(mask.mean()) if len(mask) else 0.0,
                    total_speech=float(sum(e - s for s, e in segs)),
                    segments=[[float(s), float(e)] for s, e in segs],
                    energy_db=float(stats["rms_db"]),
                    confidence=float(np.clip(snr["snr_db"] / 30.0, 0.0, 1.0)),
                )
            )
        return tracks
