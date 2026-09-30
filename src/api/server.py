"""FastAPI backend + static frontend host.

    python run.py serve            # -> http://127.0.0.1:8000

Endpoints
---------
GET  /                          the web UI
GET  /api/health                sanity + environment summary
GET  /api/methods               every denoiser / separator with availability
GET  /api/paths                 preset pipeline paths with availability
POST /api/jobs                  multipart upload -> starts a job
GET  /api/jobs                  recent jobs
GET  /api/jobs/{id}             status + events (+ report when finished)
GET  /api/jobs/{id}/events      incremental events (poll with ?since=)
GET  /api/jobs/{id}/report      the full json report
GET  /api/jobs/{id}/zip         every produced wav in one archive
GET  /api/files/{id}/{rel}      stream a produced wav
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
import zipfile
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.audio_io import AUDIO_EXTS, SUPPORTED_EXTS, VIDEO_EXTS, ffmpeg_exe, ffmpeg_available
from ..core.registry import list_denoisers, list_separators, list_transcribers, load_all
from ..core.utils import FRONTEND_DIR, OUT_DIR, ensure_dirs, human_size, jsonable, module_available
from ..core.utils import contained_path, validate_component
from ..pipeline import PipelineOptions, list_paths, resolve_path
from .jobs import MANAGER

# Uploads stream straight to disk, so a big video costs disk, not RAM.
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "1024"))
UPLOAD_CHUNK = 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    load_all()  # import every backend module so the registry is populated
    yield
    MANAGER.shutdown()


app = FastAPI(
    title="Denoise & Separate",
    version=__version__,
    description="Modular speech denoising + speaker separation + multilingual speech-to-text with a comparison UI.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  meta
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    denoisers = list_denoisers()
    separators = list_separators()
    return {
        "ok": True,
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": module_available("torch"),
        "denoisers_available": sum(1 for d in denoisers if d.available),
        "denoisers_total": len(denoisers),
        "separators_available": sum(1 for s in separators if s.available),
        "separators_total": len(separators),
        'transcribers_available': sum(m.available for m in list_transcribers()),
        'transcribers_total': len(list_transcribers()),
        "max_upload_mb": MAX_UPLOAD_MB,
        "supported_formats": sorted(SUPPORTED_EXTS),
        "audio_formats": sorted(AUDIO_EXTS),
        "video_formats": sorted(VIDEO_EXTS),
        "ffmpeg": bool(ffmpeg_available()),
        "ffmpeg_path": ffmpeg_exe() or "",
        "video_supported": bool(ffmpeg_available()),
        "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
    }


@app.get("/api/methods")
def methods() -> Dict[str, Any]:
    from ..denoising.methods import IMPORT_ERRORS as DEN_ERRORS
    from ..separation.methods import IMPORT_ERRORS as SEP_ERRORS
    from ..transcription.methods import IMPORT_ERRORS as STT_ERRORS
    from ..transcription.languages import language_options

    return jsonable(
        {
            "denoisers": [m.to_dict() for m in list_denoisers()],
            "separators": [m.to_dict() for m in list_separators()],
            'transcribers': [m.to_dict() for m in list_transcribers()],
            'languages': language_options(),
            "import_errors": {"denoising": DEN_ERRORS, "separation": SEP_ERRORS, 'transcription': STT_ERRORS},
        }
    )


@app.get("/api/paths")
def paths() -> Dict[str, Any]:
    return jsonable({"paths": list_paths()})


# --------------------------------------------------------------------------- #
#  jobs
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    paths: str = Form("[]"),
    options: str = Form("{}"),
) -> Dict[str, Any]:
    try:
        path_specs = json.loads(paths)
        if not isinstance(path_specs, list):
            raise ValueError("`paths` must be a JSON array")
        if len(path_specs) > 32:
            raise ValueError("at most 32 paths may run in a job")
        path_specs = path_specs or [{"id": "path1"}]
        opts_raw = json.loads(options)
        if not isinstance(opts_raw, dict):
            raise ValueError("`options` must be a JSON object")
        validated_options = PipelineOptions.from_dict(opts_raw)
        resolved = [resolve_path(spec if isinstance(spec, dict) else {"id": spec}) for spec in path_specs]
        if len({p.id.casefold() for p in resolved}) != len(resolved):
            raise ValueError("pipeline path ids must be unique")
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=415,
            detail="'%s' is not a format we can read. Audio: %s. Video: %s."
            % (suffix, ", ".join(sorted(AUDIO_EXTS)), ", ".join(sorted(VIDEO_EXTS))),
        )
    if suffix in VIDEO_EXTS and not ffmpeg_available():
        raise HTTPException(
            status_code=415,
            detail="Reading the audio out of a video needs ffmpeg, which is not installed. "
                   "Run: pip install imageio-ffmpeg",
        )

    # Stream to disk in 1 MB chunks: a 1 GB screen recording must not be read
    # into memory just to be written straight back out.
    saved = MANAGER.upload_target(file.filename or "upload.wav")
    total = 0
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    try:
        with saved.open("wb") as fh:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="file is larger than the %.0f MB limit (set MAX_UPLOAD_MB to raise it)"
                        % MAX_UPLOAD_MB,
                    )
                fh.write(chunk)
    except HTTPException:
        saved.unlink(missing_ok=True)
        raise
    except Exception:
        saved.unlink(missing_ok=True)
        raise

    if total == 0:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="the upload was empty")

    try:
        job = MANAGER.submit(saved, file.filename or saved.name, resolved, validated_options)
    except Exception:
        saved.unlink(missing_ok=True)
        raise
    return {
        "job_id": job.id,
        "status": job.status,
        "paths": job.paths,
        "size": human_size(total),
        "is_video": suffix in VIDEO_EXTS,
    }


@app.get("/api/jobs")
def list_jobs(limit: int = Query(30, ge=1, le=200)) -> Dict[str, Any]:
    return {"jobs": MANAGER.list(limit)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = Query(0, ge=0)) -> Dict[str, Any]:
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot(since=since, include_report=True)


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, since: int = Query(0, ge=0)) -> Dict[str, Any]:
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot(since=since, include_report=False)


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str) -> Dict[str, Any]:
    job = MANAGER.get(job_id)
    if job and job.report:
        return job.report
    stored = _safe_job_dir(job_id) / "report.json"
    if stored.exists():
        return json.loads(stored.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="no report for this job (yet)")


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str, path_id: str = Query("", description="only this path's outputs")):
    """Everything the job produced, or just one path's outputs."""
    job_dir = _safe_job_dir(job_id)

    root = job_dir
    label = job_id
    if path_id:
        if "/" in path_id or "\\" in path_id or ".." in path_id:
            raise HTTPException(status_code=400, detail="bad path id")
        root = _safe_target(job_dir, path_id)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail="no outputs for path '%s'" % path_id)
        label = "%s_%s" % (job_id, path_id)

    buf = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                _safe_target(job_dir, str(file.relative_to(job_dir)))
                zf.write(file, file.relative_to(root if path_id else job_dir))
        # a single-path zip still ships the original, so the folder is self-contained
        if path_id:
            for original in sorted(job_dir.glob("*_original.wav")):
                zf.write(original, original.name)
    buf.seek(0)
    def chunks():
        try:
            while chunk := buf.read(1024 * 1024):
                yield chunk
        finally:
            buf.close()
    return StreamingResponse(
        chunks(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="%s.zip"' % label},
    )


# --------------------------------------------------------------------------- #
#  produced files
# --------------------------------------------------------------------------- #
@app.get("/api/files/{job_id}/{rel_path:path}")
def get_file(job_id: str, rel_path: str):
    job_dir = _safe_job_dir(job_id)
    target = _safe_target(job_dir, rel_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    media = {'.wav': 'audio/wav', '.txt': 'text/plain; charset=utf-8',
             '.json': 'application/json'}.get(target.suffix.lower(), 'application/octet-stream')
    return FileResponse(str(target), media_type=media, filename=target.name)


def _safe_job_dir(job_id: str) -> Path:
    try:
        validate_component(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="bad job id") from exc
    job_dir = _safe_target(OUT_DIR, job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="unknown job")
    return job_dir


def _safe_target(root: Path, relative: str) -> Path:
    try:
        return contained_path(root, relative)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail="path traversal rejected") from exc


# --------------------------------------------------------------------------- #
#  frontend
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the UI with cache-busted asset links.

    Without this, editing styles.css / app.js and reloading shows the *old*
    file: the browser has them cached and nothing in the URL changed. Stamping
    the files' mtimes into the query string fixes that permanently.
    """
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>frontend/index.html is missing</h1>", status_code=500)

    html = page.read_text(encoding="utf-8")
    stamp = 0
    for asset in ("styles.css", "app.js"):
        target = FRONTEND_DIR / asset
        if target.exists():
            stamp = max(stamp, int(target.stat().st_mtime))
    version = "%s-%d" % (__version__, stamp)
    html = html.replace("/static/styles.css", "/static/styles.css?v=%s" % version)
    html = html.replace("/static/app.js", "/static/app.js?v=%s" % version)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.exception_handler(404)
def not_found(_request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": getattr(exc, "detail", "not found")})
