"""End-to-end check of the running server: every endpoint, every failure mode.

    python run.py serve                 # in one terminal
    python scripts/selftest.py          # in another

    python scripts/selftest.py --url http://127.0.0.1:8099 --keep

It uploads real audio (and a real video, if ffmpeg is present), waits for the
jobs, and verifies the produced files actually decode. Anything that only
"looks" right in JSON is opened and checked.

Exit code is 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.audio_io import ffmpeg_exe, load_audio, save_audio  # noqa: E402
from src.core.types import AudioBuffer  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %-52s %s" % (PASS if ok else FAIL, name, detail if not ok else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print("\n" + "-" * 78)
    print(" " + title)
    print("-" * 78)


# --------------------------------------------------------------------------- #
def make_test_audio(path: Path, seconds: float = 12.0, speakers: int = 3) -> Path:
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_audio import build_conversation

    noisy, _, _ = build_conversation(speakers, seconds, 16000, 8.0, 7)
    save_audio(path, AudioBuffer(noisy, 16000))
    return path


def make_test_video(path: Path, audio: Path) -> Optional[Path]:
    exe = ffmpeg_exe()
    if not exe:
        return None
    import subprocess

    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
           "-i", str(audio), "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    return path if proc.returncode == 0 and path.exists() else None


def wait_for_job(base: str, job_id: str, timeout: float = 600.0) -> Dict[str, Any]:
    started = time.time()
    seen_artifacts = 0
    while time.time() - started < timeout:
        r = requests.get("%s/api/jobs/%s" % (base, job_id), timeout=30)
        r.raise_for_status()
        data = r.json()
        seen_artifacts += sum(1 for e in data.get("events", []) if e.get("artifact"))
        if data["status"] in ("done", "error"):
            data["_artifact_events"] = seen_artifacts
            return data
        time.sleep(0.4)
    raise TimeoutError("job %s did not finish in %.0fs" % (job_id, timeout))


def submit(base: str, file_path: Path, paths: List[Dict[str, Any]], options: Dict[str, Any]):
    with file_path.open("rb") as fh:
        return requests.post(
            "%s/api/jobs" % base,
            files={"file": (file_path.name, fh, "application/octet-stream")},
            data={"paths": json.dumps(paths), "options": json.dumps(options)},
            timeout=300,
        )


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--keep", action="store_true", help="keep the temporary test media")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    tmp = ROOT / "data" / "cache" / "selftest"
    tmp.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" self test against %s" % base)
    print("=" * 78)

    # ---------------------------------------------------------------- meta --
    section("meta endpoints")
    try:
        health = requests.get("%s/api/health" % base, timeout=20).json()
    except Exception as exc:
        print("  cannot reach the server: %s" % exc)
        print("  start it with:  python run.py serve --port %s"
              % base.rsplit(":", 1)[-1])
        return 2

    check("GET /api/health", health.get("ok") is True)
    check("health reports formats", len(health.get("supported_formats", [])) > 10)
    check("health reports video formats", len(health.get("video_formats", [])) > 5)
    check("health reports ffmpeg flag", "ffmpeg" in health)
    check("index page serves", requests.get(base + "/", timeout=20).status_code == 200)

    page = requests.get(base + "/", timeout=20).text
    check("index cache-busts assets", "styles.css?v=" in page and "app.js?v=" in page)
    check("static css serves", requests.get(base + "/static/styles.css", timeout=20).status_code == 200)
    check("static js serves", requests.get(base + "/static/app.js", timeout=20).status_code == 200)

    methods = requests.get("%s/api/methods" % base, timeout=20).json()
    dens, seps = methods["denoisers"], methods["separators"]
    check("methods: 10 denoisers", len(dens) == 10, "got %d" % len(dens))
    check("methods: 9 separators", len(seps) == 9, "got %d" % len(seps))
    check("every method explains itself",
          all(m.get("how_it_works") and m.get("steps") for m in dens + seps),
          "missing: %s" % [m["key"] for m in dens + seps if not m.get("how_it_works")])
    check("unavailable methods say why",
          all(m["unavailable_reason"] for m in dens + seps if not m["available"]))
    check("no module import errors",
          not methods["import_errors"]["denoising"] and not methods["import_errors"]["separation"],
          str(methods["import_errors"]))

    paths = requests.get("%s/api/paths" % base, timeout=20).json()["paths"]
    check("paths: 9 presets", len(paths) == 9, "got %d" % len(paths))
    check("every path has detail",
          all(p.get("detail") and p.get("best_for") and p.get("output") for p in paths))
    check("blocked paths carry an install hint",
          all(p["install_hints"] for p in paths if not p["available"]))
    check("blockers are de-duplicated",
          all(len(p["blockers"]) == len(set(p["blockers"])) for p in paths))

    ready = [p["id"] for p in paths if p["available"]]
    check("at least one path is ready", bool(ready), "none available")
    if not ready:
        return 1
    print("  ready paths: %s" % ", ".join(ready))

    # ------------------------------------------------------------- rejects --
    section("input validation (should be refused politely)")
    bad = tmp / "notes.txt"
    bad.write_text("this is not audio", encoding="utf-8")
    r = submit(base, bad, [{"id": ready[0]}], {})
    check("unsupported extension -> 415", r.status_code == 415, "got %d" % r.status_code)
    check("415 explains the formats", "video" in r.text.lower() or "audio" in r.text.lower())

    empty = tmp / "empty.wav"
    empty.write_bytes(b"")
    r = submit(base, empty, [{"id": ready[0]}], {})
    check("empty upload -> 4xx", 400 <= r.status_code < 500, "got %d" % r.status_code)

    r = requests.get("%s/api/jobs/does-not-exist" % base, timeout=20)
    check("unknown job -> 404", r.status_code == 404, "got %d" % r.status_code)

    r = requests.get("%s/api/files/..%%2F..%%2Fsecret.txt/x.wav" % base, timeout=20)
    check("path traversal refused", r.status_code in (400, 404), "got %d" % r.status_code)

    # ------------------------------------------------------------ real run --
    section("audio job (multi-path)")
    audio = make_test_audio(tmp / "selftest_audio.wav", seconds=12.0, speakers=3)
    run_paths = [{"id": pid} for pid in ready[:3]]
    r = submit(base, audio, run_paths, {"num_speakers": 3})
    check("POST /api/jobs accepted", r.status_code == 200, r.text[:120])
    if r.status_code != 200:
        return 1
    job_id = r.json()["job_id"]

    job = wait_for_job(base, job_id)
    check("job finished", job["status"] == "done", str(job.get("error"))[:120])
    check("progress reached 100%", job["pct"] >= 0.999)
    check("artifacts streamed during the run", job.get("_artifact_events", 0) > 0,
          "results should appear one by one")
    check("no traceback leaked into events",
          not any("Traceback" in str(e.get("message", "")) for e in job.get("events", [])))

    report = job.get("report") or {}
    check("report has an input block", bool(report.get("input", {}).get("duration")))
    check("report has one entry per path", len(report.get("paths", [])) == len(run_paths))
    check("comparison was built", bool(report.get("comparison", {}).get("ranking")))

    ok_paths = [p for p in report.get("paths", []) if p["status"] == "ok"]
    check("all requested paths succeeded", len(ok_paths) == len(run_paths),
          "%d/%d" % (len(ok_paths), len(run_paths)))

    # ---- the files themselves ---------------------------------------- #
    section("produced files")
    all_names: List[str] = []
    for p in ok_paths:
        den = p.get("denoised", {})
        url = "%s/api/files/%s/%s/%s" % (base, job_id, p["id"], den.get("file", ""))
        resp = requests.get(url, timeout=60)
        ok = resp.status_code == 200 and resp.content[:4] == b"RIFF"
        check("%s: denoised file downloads and is a wav" % p["id"], ok)
        all_names.append(den.get("file", ""))

        tracks = p.get("tracks", [])
        check("%s: produced %d speaker file(s)" % (p["id"], len(tracks)), len(tracks) >= 1)
        for t in tracks:
            turl = "%s/api/files/%s/%s/%s" % (base, job_id, p["id"], t["file"])
            tr = requests.get(turl, timeout=60)
            if tr.status_code != 200 or tr.content[:4] != b"RIFF":
                check("%s: %s downloads" % (p["id"], t["file"]), False)
                break
            with wave.open(io.BytesIO(tr.content), "rb") as wf:
                frames = wf.getnframes()
            if frames <= 0:
                check("%s: %s has audio" % (p["id"], t["file"]), False)
                break
            all_names.append(t["file"])
        else:
            check("%s: every speaker file decodes" % p["id"], True)

    check("output names are unique across paths",
          len(all_names) == len(set(all_names)),
          "collisions: %s" % [n for n in all_names if all_names.count(n) > 1])
    check("names carry the source file and the path",
          all("selftest_audio" in n and any(p["id"] in n for p in ok_paths) for n in all_names if n),
          "example: %s" % (all_names[0] if all_names else "-"))

    # ---- zips --------------------------------------------------------- #
    section("downloads")
    z = requests.get("%s/api/jobs/%s/zip" % (base, job_id), timeout=120)
    check("full zip downloads", z.status_code == 200 and z.content[:2] == b"PK")
    if z.status_code == 200:
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        check("full zip holds every path", all(any(p["id"] in n for n in names) for p in ok_paths))
        check("full zip holds the original", any("original" in n for n in names))

    one = requests.get("%s/api/jobs/%s/zip?path_id=%s" % (base, job_id, ok_paths[0]["id"]), timeout=120)
    check("per-path zip downloads", one.status_code == 200 and one.content[:2] == b"PK")
    if one.status_code == 200:
        names = zipfile.ZipFile(io.BytesIO(one.content)).namelist()
        others = [p["id"] for p in ok_paths[1:]]
        check("per-path zip holds only that path",
              not any(any(o in n for o in others) for n in names), str(names))

    check("report endpoint serves",
          requests.get("%s/api/jobs/%s/report" % (base, job_id), timeout=30).status_code == 200)
    check("job list includes this job",
          any(j["job_id"] == job_id for j in requests.get("%s/api/jobs" % base, timeout=30).json()["jobs"]))

    # ------------------------------------------------------------- video --
    section("video input")
    if not ffmpeg_exe():
        check("ffmpeg present (video support)", False, "install: pip install imageio-ffmpeg")
    else:
        video = make_test_video(tmp / "selftest_video.mp4", audio)
        if not video:
            check("test video built", False)
        else:
            r = submit(base, video, [{"id": ready[0]}], {"num_speakers": 3})
            check("video upload accepted", r.status_code == 200, r.text[:120])
            if r.status_code == 200:
                vjob = wait_for_job(base, r.json()["job_id"])
                check("video job finished", vjob["status"] == "done", str(vjob.get("error"))[:120])
                vrep = vjob.get("report", {})
                dur = vrep.get("input", {}).get("duration", 0)
                check("audio track extracted from the video", dur > 5, "duration %.1fs" % dur)
                vok = [p for p in vrep.get("paths", []) if p["status"] == "ok"]
                check("video produced speaker files", bool(vok) and bool(vok[0].get("tracks")))

    # ------------------------------------------------- friendly failures --
    section("failure handling (must be human-readable, never a traceback)")
    blocked = [p for p in paths if not p["available"]]
    if blocked:
        target = blocked[0]
        r = submit(base, audio, [{"id": target["id"]}], {})
        if check("blocked path is still accepted for a run", r.status_code == 200):
            bjob = wait_for_job(base, r.json()["job_id"])
            brep = bjob.get("report", {})
            bpaths = brep.get("paths", [])
            failed = [p for p in bpaths if p["status"] == "failed"]
            check("blocked path reports failure", bool(failed))
            if failed:
                err = str(failed[0].get("error", ""))
                check("error has no traceback",
                      "Traceback" not in err and ".py" not in err and "line " not in err, err[:100])
                check("error is a readable sentence", len(err) > 10 and err[0].isupper(), err[:100])
                check("error suggests a fix", bool(failed[0].get("error_fix")), "no fix offered")
    else:
        print("  (every path is installed -- nothing to fail)")

    if not args.keep:
        for f in (tmp / "selftest_audio.wav", tmp / "selftest_video.mp4", bad, empty):
            try:
                f.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------ verdict --
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print("\n" + "=" * 78)
    print(" %d/%d checks passed" % (passed, total))
    if passed != total:
        print("\n failures:")
        for r in results:
            if not r["ok"]:
                print("   - %s  %s" % (r["name"], r["detail"]))
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
