"""In-process job manager: one upload -> a background worker -> a growing report.

Deliberately dependency-free (threads + a dict), because the whole point of this
project is that it runs from ``python run.py serve`` with no broker, no redis and
no docker.  Swapping this for Celery/RQ later only means reimplementing
:class:`JobManager`.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.errors import describe_exception, log_exception
from ..core.utils import RAW_DIR, jsonable, new_id, safe_name
from ..pipeline import PipelineOptions, PipelineRunner, PipelinePath

MAX_EVENTS = 4000


@dataclass
class Job:
    id: str
    filename: str
    input_path: str
    paths: List[str]
    status: str = "queued"  # queued | running | done | error
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    pct: float = 0.0
    message: str = "queued"
    events: List[Dict[str, Any]] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_fix: str = ""
    _next_seq: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            event = dict(event)
            event["seq"] = self._next_seq
            self._next_seq += 1
            self.events.append(event)
            if len(self.events) > MAX_EVENTS:
                del self.events[: len(self.events) - MAX_EVENTS]
            if "pct" in event:
                self.pct = max(self.pct, float(event["pct"]))
            if event.get("message"):
                self.message = str(event["message"])

    def snapshot(self, since: int = 0, include_report: bool = True) -> Dict[str, Any]:
        with self._lock:
            events = [e for e in self.events if e.get("seq", 0) >= since]
            data = {
                "job_id": self.id,
                "filename": self.filename,
                "status": self.status,
                "pct": round(self.pct, 4),
                "message": self.message,
                "created": self.created,
                "started": self.started,
                "finished": self.finished,
                "elapsed": round((self.finished or time.time()) - (self.started or self.created), 2),
                "paths": self.paths,
                "events": events,
                "next_seq": self._next_seq,
                "error": self.error,
                "error_fix": self.error_fix,
            }
            if include_report and self.report is not None:
                data["report"] = self.report
            return jsonable(data)


class JobManager:
    def __init__(self, workers: int = 2) -> None:
        self._jobs: Dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pipeline")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def upload_target(self, filename: str) -> Path:
        """Where an upload should be written (the caller streams into it)."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        name = "%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex, safe_name(filename))
        return RAW_DIR / name

    def save_upload(self, filename: str, data: bytes) -> Path:
        target = self.upload_target(filename)
        target.write_bytes(data)
        return target

    # ------------------------------------------------------------------ #
    def submit(
        self,
        input_path: Path,
        filename: str,
        paths: List[PipelinePath],
        options: PipelineOptions,
    ) -> Job:
        job = Job(
            id=new_id("job"),
            filename=filename,
            input_path=str(input_path),
            paths=[p.id for p in paths],
        )
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, input_path, paths, options)
        return job

    def _run(self, job: Job, input_path: Path, paths: List[PipelinePath], options: PipelineOptions) -> None:
        job.status = "running"
        job.started = time.time()
        job.add_event({"stage": "queued", "pct": 0.01, "message": "worker picked up the job", "ts": time.time()})
        try:
            runner = PipelineRunner(job_id=job.id, progress=job.add_event)
            job.report = runner.run(input_path, paths, options)
            job.status = "done"
            job.pct = 1.0
            job.message = "finished"
        except Exception as exc:
            # The traceback goes to the server log; the browser gets a sentence.
            log_exception("job:%s" % job.id, exc, {"file": job.filename, "paths": job.paths})
            info = describe_exception(exc)
            job.status = "error"
            job.error = info["message"]
            job.error_fix = info["fix"]
            job.add_event({"stage": "error", "pct": 1.0, "message": info["message"],
                           "fix": info["fix"], "ts": time.time()})
        finally:
            job.finished = time.time()

    # ------------------------------------------------------------------ #
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.created)[:limit]
        return [
            {
                "job_id": j.id,
                "filename": j.filename,
                "status": j.status,
                "pct": round(j.pct, 3),
                "created": j.created,
                "paths": j.paths,
            }
            for j in jobs
        ]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


MANAGER = JobManager()
