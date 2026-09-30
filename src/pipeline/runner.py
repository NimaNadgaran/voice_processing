"""The orchestrator: upload -> denoise -> count speakers -> separate -> report.

One :class:`PipelineRunner` handles one job, which may execute several *paths*
over the same input so the frontend can compare them.

Design decision worth knowing about
-----------------------------------
The speaker count is estimated on the **original** audio, then pinned for the
separator that runs on the **denoised** audio.  Measured on the bundled demo
(4 speakers, 7 dB SNR): counting on the raw signal found 4/4 speakers, counting
after aggressive gating found only 2 -- strong denoising strips exactly the
low-level spectral detail that speaker embeddings rely on, while separation
still benefits from the cleaner signal.  Switch it with
``PipelineOptions(count_on="denoised")`` if your denoiser is gentle.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ..core.audio_io import load_audio, save_audio, waveform_preview
from ..core.errors import AppError, describe_exception, log_exception
from ..core.metrics import estimated_snr, fun_facts, signal_stats, spectral_profile
from ..core.types import AudioBuffer
from ..core.utils import OUT_DIR, Timer, human_time, jsonable, new_id, safe_name, write_json
from ..denoising import denoise as run_denoise
from ..separation import separate as run_separate
from ..separation.speaker_count import estimate_speaker_count
from .paths import PipelinePath, availability

ProgressFn = Optional[Callable[[Dict[str, Any]], None]]

MAX_STEM = 40


#: uploads are stored as "<YYYYmmdd-HHMMSS>_<original name>"; that prefix is
#: useful on disk but redundant inside a job folder, so strip it for output names
_UPLOAD_STAMP = re.compile(r"^\d{8}-\d{6}_(?:[a-f0-9]{32}_)?")


def source_stem(filename: str) -> str:
    """A short, filesystem-safe stem taken from the uploaded file's name."""
    stem = safe_name(Path(filename).stem, fallback="audio")
    stem = _UPLOAD_STAMP.sub("", stem)
    return stem[:MAX_STEM].strip("._-") or "audio"


# --------------------------------------------------------------------------- #
class PipelineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    num_speakers: Optional[int] = Field(default=None, ge=1, le=32)
    max_speakers: int = Field(default=8, ge=1, le=32)
    count_on: Literal["original", "denoised"] = "original"
    save_waveforms: bool = True
    waveform_points: int = Field(default=500, ge=1, le=10000)
    normalize_tracks: bool = False
    keep_intermediate: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PipelineOptions":
        return cls.model_validate({} if data is None else data)


# --------------------------------------------------------------------------- #
class PipelineRunner:
    """Runs one or more paths over a single input file and writes a report."""

    def __init__(
        self,
        job_id: Optional[str] = None,
        out_root: Optional[Path] = None,
        progress: ProgressFn = None,
    ) -> None:
        self.job_id = job_id or new_id("job")
        self.out_dir = Path(out_root or OUT_DIR) / self.job_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._progress = progress
        self.events: List[Dict[str, Any]] = []
        self.stem = "audio"  # replaced with the upload's name in run()

    # ------------------------------------------------------------------ #
    def emit(self, **event: Any) -> None:
        event.setdefault("ts", time.time())
        self.events.append(event)
        if self._progress:
            try:
                self._progress(event)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def run(
        self,
        audio_path: Path,
        paths: List[PipelinePath],
        options: Optional[PipelineOptions] = None,
    ) -> Dict[str, Any]:
        options = options or PipelineOptions()
        if len({p.id.casefold() for p in paths}) != len(paths):
            raise ValueError("pipeline path ids must be unique")
        started = time.time()

        self.emit(stage="load", pct=0.02, message="reading %s" % Path(audio_path).name)
        with Timer() as t_load:
            audio = load_audio(audio_path)

        # Every produced file is named after the *source* file and the path that
        # made it. Two paths otherwise both emit "speaker_01.wav", which collide
        # the moment you download them into the same folder.
        self.stem = source_stem(Path(audio_path).name)
        original_copy = self.out_dir / ("%s_original.wav" % self.stem)
        save_audio(original_copy, audio)

        input_info = {
            "filename": Path(audio_path).name,
            "path": str(original_copy),
            "duration": round(audio.duration, 3),
            "sample_rate": audio.sr,
            "samples": audio.n_samples,
            "stats": signal_stats(audio),
            "snr": estimated_snr(audio),
            "spectrum": spectral_profile(audio),
            "waveform": waveform_preview(audio, options.waveform_points) if options.save_waveforms else [],
            "load_seconds": round(t_load.elapsed, 3),
        }
        self.emit(stage="load", pct=0.06,
                  message="loaded %.1fs @ %d Hz (estimated SNR %.1f dB)"
                          % (audio.duration, audio.sr, input_info["snr"]["snr_db"]))

        # ---- speaker count on the ORIGINAL signal ----------------------- #
        speaker_estimate: Dict[str, Any] = {}
        if options.num_speakers:
            speaker_estimate = {
                "n_speakers": int(options.num_speakers),
                "confidence": 1.0,
                "method": "user-specified",
                "source": "user",
            }
            self.emit(stage="count", pct=0.1,
                      message="speaker count pinned by user: %d" % options.num_speakers)
        elif options.count_on == "original":
            self.emit(stage="count", pct=0.08, message="estimating how many people are talking")
            with Timer() as t_count:
                est = estimate_speaker_count(audio, max_k=options.max_speakers)
            speaker_estimate = {
                "n_speakers": int(est["n_speakers"]),
                "confidence": float(est["confidence"]),
                "method": est["method"],
                "scores": est["scores"],
                "eigengap_k": est.get("eigengap_k"),
                "n_windows": est.get("n_windows"),
                "source": "original",
                "seconds": round(t_count.elapsed, 3),
            }
            self.emit(stage="count", pct=0.12,
                      message="estimated %d speaker(s), confidence %.0f%%"
                              % (speaker_estimate["n_speakers"], 100 * speaker_estimate["confidence"]))

        # ---- run each path ---------------------------------------------- #
        results: List[Dict[str, Any]] = []
        n_paths = max(1, len(paths))
        for i, path in enumerate(paths):
            base = 0.15 + 0.8 * (i / n_paths)
            span = 0.8 / n_paths
            self.emit(stage="path", path_id=path.id, pct=base,
                      message="path %d/%d -- %s" % (i + 1, n_paths, path.name))
            results.append(self._run_one(audio, path, options, speaker_estimate, base, span))

        report = {
            "job_id": self.job_id,
            "created": started,
            "elapsed": round(time.time() - started, 3),
            "elapsed_human": human_time(time.time() - started),
            "input": input_info,
            "speaker_estimate": speaker_estimate,
            "options": {
                "num_speakers": options.num_speakers,
                "max_speakers": options.max_speakers,
                "count_on": options.count_on,
            },
            "paths": results,
            "out_dir": str(self.out_dir),
        }

        from .comparison import build_comparison

        report["comparison"] = build_comparison(report)
        write_json(self.out_dir / "report.json", report)
        self.emit(stage="done", pct=1.0,
                  message="finished %d path(s) in %s" % (len(results), human_time(report["elapsed"])))
        return jsonable(report)

    # ------------------------------------------------------------------ #
    def _run_one(
        self,
        audio: AudioBuffer,
        path: PipelinePath,
        options: PipelineOptions,
        speaker_estimate: Dict[str, Any],
        base: float,
        span: float,
    ) -> Dict[str, Any]:
        path_dir = self.out_dir / path.id
        path_dir.mkdir(parents=True, exist_ok=True)

        # Swap in stand-ins for any backend that is not installable here, so the
        # rest of this method -- filenames included -- talks about what actually
        # ran.  The swaps themselves are kept in the record for the report.
        substitutions = availability(path)["substitutions"]
        path = path.resolved()

        record: Dict[str, Any] = {
            "id": path.id,
            "name": path.name,
            "tagline": path.tagline,
            "badge": path.badge,
            "denoiser": path.denoiser,
            "separator": path.separator,
            "substitutions": substitutions,
            "status": "running",
            "timings": {},
            "logs": [],
            "out_dir": str(path_dir),
        }
        record.update({k: v for k, v in availability(path).items() if k.endswith("_name")})

        total = Timer()
        total.__enter__()

        # ---------------- denoise ---------------------------------------- #
        try:
            self.emit(stage="denoise", path_id=path.id, pct=base + 0.05 * span,
                      message="denoising with %s" % record.get("denoiser_name", path.denoiser))

            def d_progress(pct: float, msg: str) -> None:
                self.emit(stage="denoise", path_id=path.id,
                          pct=base + span * (0.05 + 0.35 * pct), message=msg)

            den = run_denoise(audio, method=path.denoiser, progress=d_progress, safe=True)
            if den.metrics.get("error"):
                raise AppError(den.metrics["error"],
                               fix=den.metrics.get("error_fix", ""))

            den_path = path_dir / ("%s__%s__01_denoised_%s.wav" % (self.stem, path.id, path.denoiser))
            save_audio(den_path, den.audio)
            record["denoised"] = {
                "path": str(den_path),
                "file": den_path.name,
                "method": path.denoiser,
                "method_name": record.get("denoiser_name", path.denoiser),
                "backend": den.backend_used,
                "elapsed": round(den.elapsed, 3),
                "metrics": den.metrics,
                "stats": signal_stats(den.audio),
                "spectrum": spectral_profile(den.audio),
                "waveform": waveform_preview(den.audio, options.waveform_points) if options.save_waveforms else [],
            }
            record["timings"]["denoise"] = round(den.elapsed, 3)
            record["logs"].extend(den.logs)
            self.emit(stage="denoise", path_id=path.id, pct=base + 0.4 * span,
                      artifact={"kind": "denoised", "path_id": path.id, "file": den_path.name},
                      message="denoised in %s (SNR %+.1f dB)"
                              % (human_time(den.elapsed), den.metrics.get("snr_improvement_db", 0.0)))
            clean = den.audio
        except Exception as exc:
            log_exception("denoise:%s" % path.denoiser, exc, {"path": path.id})
            info = describe_exception(exc, "denoising")
            record["status"] = "failed"
            record["error"] = info["message"]
            record["error_fix"] = info["fix"]
            record["failed_stage"] = "denoise"
            total.__exit__()
            record["timings"]["total"] = round(total.elapsed, 3)
            self.emit(stage="error", path_id=path.id, pct=base + span,
                      message=info["message"], fix=info["fix"])
            return record

        # ---------------- optional re-count on the denoised signal ------- #
        n_speakers = speaker_estimate.get("n_speakers")
        if not options.num_speakers and options.count_on == "denoised":
            est = estimate_speaker_count(clean, max_k=options.max_speakers)
            n_speakers = int(est["n_speakers"])
            record["speaker_estimate"] = {
                "n_speakers": n_speakers,
                "confidence": float(est["confidence"]),
                "method": est["method"],
                "scores": est["scores"],
                "source": "denoised",
            }

        # ---------------- separate --------------------------------------- #
        try:
            self.emit(stage="separate", path_id=path.id, pct=base + 0.45 * span,
                      message="separating with %s" % record.get("separator_name", path.separator))

            def s_progress(pct: float, msg: str) -> None:
                self.emit(stage="separate", path_id=path.id,
                          pct=base + span * (0.45 + 0.45 * pct), message=msg)

            sep = run_separate(
                clean, method=path.separator, num_speakers=n_speakers,
                progress=s_progress, safe=True,
            )
            if sep.metrics.get("error"):
                raise AppError(sep.metrics["error"],
                               fix=sep.metrics.get("error_fix", ""))

            tracks = []
            for track in sep.tracks:
                if options.normalize_tracks:
                    from ..core.audio_io import peak_normalize

                    track.audio = AudioBuffer(peak_normalize(track.audio.samples), track.audio.sr)
                file_name = "%s__%s__02_speaker%02d_%s.wav" % (
                    self.stem, path.id, track.index + 1, path.separator,
                )
                save_audio(path_dir / file_name, track.audio)
                track.path = str(path_dir / file_name)
                entry = track.to_dict()
                entry["file"] = file_name
                entry["waveform"] = (
                    waveform_preview(track.audio, options.waveform_points) if options.save_waveforms else []
                )
                entry["snr"] = estimated_snr(track.audio)
                tracks.append(entry)
                self.emit(
                    stage="separate", path_id=path.id, pct=base + 0.9 * span,
                    artifact={"kind": "speaker", "path_id": path.id, "file": file_name, "index": track.index},
                    message="wrote %s (%.1fs of speech)" % (entry["label"], entry["total_speech"]),
                )

            record["tracks"] = tracks
            record["separation"] = {
                "method": path.separator,
                "method_name": record.get("separator_name", path.separator),
                "backend": sep.backend_used,
                "elapsed": round(sep.elapsed, 3),
                "metrics": sep.metrics,
                "confidence": round(float(sep.n_speakers_confidence), 4),
            }
            record["timings"]["separate"] = round(sep.elapsed, 3)
            record["logs"].extend(sep.logs)
            record["n_speakers"] = len(tracks)
            record["fun"] = fun_facts(audio, [t.audio for t in sep.tracks])
            record["status"] = "ok"
        except Exception as exc:
            log_exception("separate:%s" % path.separator, exc, {"path": path.id})
            info = describe_exception(exc, "separation")
            record["status"] = "failed"
            record["error"] = info["message"]
            record["error_fix"] = info["fix"]
            record["failed_stage"] = "separate"
            self.emit(stage="error", path_id=path.id, pct=base + span,
                      message=info["message"], fix=info["fix"])

        total.__exit__()
        record["timings"]["total"] = round(total.elapsed, 3)
        record["timings"]["total_human"] = human_time(total.elapsed)
        if audio.duration > 0:
            record["timings"]["realtime_factor"] = round(total.elapsed / audio.duration, 3)

        write_json(path_dir / "result.json", record)
        self.emit(stage="path_done", path_id=path.id, pct=base + span,
                  message="%s finished in %s" % (path.name, human_time(total.elapsed)),
                  status=record["status"])
        return record


# --------------------------------------------------------------------------- #
def run_pipeline(
    audio_path: Path,
    paths: List[PipelinePath],
    options: Optional[PipelineOptions] = None,
    progress: ProgressFn = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper used by the CLI and the API."""
    return PipelineRunner(job_id=job_id, progress=progress).run(Path(audio_path), paths, options)
