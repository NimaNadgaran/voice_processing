#!/usr/bin/env python
"""Denoise & Separate -- single entry point.

    python run.py serve                          start the web UI (default port 8000)
    python run.py doctor                         what is installed / what is missing
    python run.py methods                        list every denoiser and separator
    python run.py paths                          list the preset pipeline paths

    python run.py denoise  in.wav --method deepfilternet -o clean.wav
    python run.py separate in.wav --method pyannote --speakers 4 -o out/
    python run.py pipeline in.wav --paths path1,path4 --speakers 4

Nothing in this file trains anything. Training lives in
``src/denoising/training`` and ``src/separation/training`` and is only ever
started by you, explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Cap the BLAS thread pools BEFORE numpy is imported.
#   * OpenBLAS allocates per-thread buffers at import time and simply gives up
#     ("Memory allocation still failed after 10 retries") in constrained
#     environments -- containers, CI, or a machine that is already busy.
#   * We also run several pipeline paths in parallel threads, so an unbounded
#     BLAS pool would oversubscribe the CPU anyway.
# Set DS_THREADS=<n> (or the standard vars) to override.
# ---------------------------------------------------------------------------
_DEFAULT_THREADS = os.environ.get("DS_THREADS") or str(max(1, min(4, os.cpu_count() or 1)))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _DEFAULT_THREADS)

from src.core.utils import ensure_dirs, human_time  # noqa: E402


# --------------------------------------------------------------------------- #
#  port helpers
#
#  A busy port is the most common way a bare `python run.py` fails, and it is
#  almost always a previous run still alive in another window (PyCharm keeps
#  them). uvicorn's own message is a WinError number, so probe first and say
#  something useful instead.
# --------------------------------------------------------------------------- #
DEFAULT_PORT = 8000
PORT_SCAN = 20  # how far past the default to look for a free port


def _port_is_free(host: str, port: int) -> bool:
    import socket

    # Deliberately no SO_REUSEADDR: on Windows that would let this test bind on
    # top of the process that already owns the port instead of reporting it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("" if host == "0.0.0.0" else host, port))
        except OSError:
            return False
    return True


def _next_free_port(host: str, start: int, tries: int = PORT_SCAN):
    for candidate in range(start + 1, start + 1 + tries):
        if _port_is_free(host, candidate):
            return candidate
    return None


def _pid_on_port(port: int) -> str:
    """Best-effort: the pid holding ``port``, or "" if we cannot tell."""
    import subprocess

    cmd = ["netstat", "-ano"] if os.name == "nt" else ["ss", "-ltnp"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout or ""
    except Exception:
        return ""

    needle = ":%d " % port  # the colon keeps :8000 from matching :18000
    for line in out.splitlines():
        if needle not in line + " ":
            continue
        if os.name == "nt":
            if "LISTENING" not in line:
                continue
            return line.split()[-1]
        if "pid=" in line:
            digits = ""
            for ch in line.split("pid=", 1)[1]:
                if not ch.isdigit():
                    break
                digits += ch
            if digits:
                return digits
    return ""


def _port_holder(host: str, port: int) -> str:
    """A human description of whoever already owns ``host:port``."""
    import json as _json
    import urllib.request

    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    ours = ""
    try:
        url = "http://%s:%d/api/health" % (probe, port)
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            body = _json.loads(resp.read().decode("utf-8", "replace"))
        if body.get("ok"):
            ours = "another Denoise & Separate (v%s)" % body.get("version", "?")
    except Exception:
        pass  # not ours, or not speaking HTTP -- the pid alone will do

    pid = _pid_on_port(port)
    if ours and pid:
        return "%s, pid %s" % (ours, pid)
    return ours or ("pid %s" % pid if pid else "another program")


# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed.  pip install -r requirements.txt")
        return 1
    ensure_dirs()

    # --port given explicitly is a request, not a suggestion: if it is taken,
    # say who has it rather than quietly moving the server somewhere else.
    explicit = args.port is not None
    port = args.port if explicit else DEFAULT_PORT
    if not _port_is_free(args.host, port):
        holder = _port_holder(args.host, port)
        pid = _pid_on_port(port)
        if explicit:
            print("  port %d on %s is already in use by %s." % (port, args.host, holder))
            print("  fix: stop it, or serve somewhere else --")
            print("         python run.py serve --port %d" % (port + 1))
            if os.name == "nt" and pid:
                print("       to stop it:  taskkill /PID %s /F" % pid)
            return 1
        free = _next_free_port(args.host, port)
        if free is None:
            print("  ports %d-%d are all in use. free one, or pass --port."
                  % (port, port + PORT_SCAN))
            return 1
        print("  port %d is taken by %s -- using %d instead."
              % (port, holder, free), flush=True)
        port = free

    # flush: PyCharm's console is a pipe, so stdout is block-buffered and this
    # line would otherwise appear after uvicorn's (stderr) logs, or not at all.
    print("\n  Denoise & Separate  ->  http://%s:%d\n" % (args.host, port), flush=True)
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    import platform

    from src.core.registry import list_denoisers, list_separators
    from src.core.utils import module_available

    print("=" * 72)
    print(" environment")
    print("=" * 72)
    print("  python      : %s" % sys.version.split()[0])
    print("  platform    : %s" % platform.platform())
    for mod in ("numpy", "scipy", "soundfile", "librosa", "torch", "fastapi", "uvicorn"):
        print("  %-11s : %s" % (mod, "yes" if module_available(mod) else "NO"))

    from src.core.audio_io import VIDEO_EXTS, ffmpeg_exe

    exe = ffmpeg_exe()
    print("  ffmpeg      : %s" % (exe if exe else "NO"))
    if exe:
        print("  video       : yes -- %s" % ", ".join(sorted(VIDEO_EXTS)))
    else:
        print("  video       : NO -- video files cannot be read")
        print("                fix: pip install imageio-ffmpeg")

    if sys.version_info >= (3, 12):
        print(
            "\n  ! heads-up: some optional backends cannot be installed on this\n"
            "    interpreter. torch itself is usually fine -- the real blockers are\n"
            "    packages still pinned to numpy<2.0 (no wheel on 3.13+) or to a\n"
            "    Rust/C extension with no matching wheel:\n"
            "        deepfilternet     numpy<2.0 + deepfilterlib (Rust, no wheel)\n"
            "        clearvoice        numpy<2.0 (tries to compile numpy 1.x)\n"
            "        resemble-enhance  source build\n"
            "    Installing them would downgrade numpy and break torch here.\n"
            "    Use a Python 3.11/3.12 venv for those; everything else works.\n"
            "    deepfilternet has a way out -- the project ships a standalone\n"
            "    binary with the weights compiled in, no python involved:\n"
            "        python scripts/download_models.py deepfilternet"
        )

    for title, items in (("denoisers", list_denoisers()), ("separators", list_separators())):
        print("\n" + "=" * 72)
        print(" %s" % title)
        print("=" * 72)
        for m in items:
            mark = "OK " if m.available else "-- "
            print("  %s %-24s %s" % (mark, m.key, m.name))
            if not m.available:
                print("        %s" % m.unavailable_reason)
                if m.install_hint:
                    print("        fix: %s" % m.install_hint)
    return 0


def cmd_methods(args: argparse.Namespace) -> int:
    from src.core.registry import list_denoisers, list_separators

    payload = {
        "denoisers": [m.to_dict() for m in list_denoisers()],
        "separators": [m.to_dict() for m in list_separators()],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for title in ("denoisers", "separators"):
        print("\n%s" % title.upper())
        for m in payload[title]:
            print("  %-24s %-34s %s %s"
                  % (m["key"], m["name"], "[available]" if m["available"] else "[missing]",
                     "" if m["available"] else "-- " + m["unavailable_reason"]))
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    from src.pipeline import list_paths

    items = list_paths()
    if args.json:
        print(json.dumps(items, indent=2))
        return 0
    for p in items:
        print("\n%-8s %s  %s" % (p["id"], p["name"], "[ready]" if p["available"] else "[blocked]"))
        print("         %s" % p["tagline"])
        for sub in p.get("substitutions", []):
            print("         ~ %s -> %s (%s)" % (sub["from_name"], sub["to_name"], sub["reason"]))
        for blocker in p["blockers"]:
            print("         ! %s" % blocker)
    return 0


def cmd_denoise(args: argparse.Namespace) -> int:
    from src.denoising import denoise

    out = Path(args.output or (Path(args.input).stem + "_denoised.wav"))
    res = denoise(args.input, method=args.method, output_path=out,
                  progress=lambda p, m: print("  [%3.0f%%] %s" % (p * 100, m)))
    if res.metrics.get("error"):
        print("FAILED: %s" % res.metrics["error"])
        return 1
    print("\nwrote %s in %s" % (out, human_time(res.elapsed)))
    print(json.dumps(res.metrics, indent=2))
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    from src.separation import separate

    out_dir = Path(args.output or (Path(args.input).stem + "_speakers"))
    res = separate(args.input, method=args.method, num_speakers=args.speakers,
                   output_dir=out_dir,
                   progress=lambda p, m: print("  [%3.0f%%] %s" % (p * 100, m)))
    if res.metrics.get("error"):
        print("FAILED: %s" % res.metrics["error"])
        return 1
    print("\n%d speaker(s) in %s -> %s" % (res.n_speakers, human_time(res.elapsed), out_dir))
    for t in res.tracks:
        print("  %-12s %6.1fs speech  %2d turns  %s"
              % (t.label, t.total_speech, len(t.segments), Path(t.path).name))
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    from src.pipeline import PipelineOptions, resolve_path, run_pipeline

    specs = [{"id": p.strip()} for p in args.paths.split(",") if p.strip()]
    if args.denoiser or args.separator:
        specs = [{"denoiser": args.denoiser or "none", "separator": args.separator or "diarize_cluster"}]
    paths = [resolve_path(s) for s in specs]

    report = run_pipeline(
        Path(args.input), paths,
        PipelineOptions(num_speakers=args.speakers, count_on=args.count_on),
        progress=lambda e: print("  [%3.0f%%] %-10s %s" % (e.get("pct", 0) * 100, e.get("stage", ""), e.get("message", ""))),
    )
    print("\n" + "=" * 72)
    print(" job %s  --  %s" % (report["job_id"], report["elapsed_human"]))
    print(" output: %s" % report["out_dir"])
    print("=" * 72)
    for p in report["paths"]:
        print("\n %-8s %-28s %s" % (p["id"], p["name"], p["status"]))
        if p.get("error"):
            print("   ! %s" % p["error"])
            continue
        d = p.get("denoised", {}).get("metrics", {})
        print("   denoise : %+.1f dB SNR, noise floor %+.1f dB, %s"
              % (d.get("snr_improvement_db", 0), -d.get("noise_reduction_db", 0),
                 human_time(p["timings"].get("denoise", 0))))
        print("   speakers: %d" % p.get("n_speakers", 0))
        for t in p.get("tracks", []):
            print("      %-12s %6.1fs speech  %s" % (t["label"], t["total_speech"], t["file"]))
    ranking = report.get("comparison", {}).get("ranking", [])
    if len(ranking) > 1:
        print("\n RANKING")
        for r in ranking:
            print("   %-8s overall %5.1f  (denoise %.0f / separate %.0f / %.2fs)"
                  % (r["id"], r["overall"], r["denoise_score"], r["separation_score"], r["seconds"]))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    import subprocess

    return subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "make_demo_audio.py"),
         "--speakers", str(args.speakers), "--seconds", str(args.seconds)]
    )


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="default %d; when left unset a busy port rolls to the next free one"
                        % DEFAULT_PORT)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="show what is installed and what is missing")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("methods", help="list denoisers and separators")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_methods)

    p = sub.add_parser("paths", help="list preset pipeline paths")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_paths)

    p = sub.add_parser("denoise", help="denoise one file")
    p.add_argument("input")
    p.add_argument("--method", default="spectral_gate")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_denoise)

    p = sub.add_parser("separate", help="split speakers in one file")
    p.add_argument("input")
    p.add_argument("--method", default="diarize_cluster")
    p.add_argument("--speakers", type=int, default=None)
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_separate)

    p = sub.add_parser("pipeline", help="denoise + separate, one or more paths")
    p.add_argument("input")
    p.add_argument("--paths", default="path1", help="comma separated preset ids")
    p.add_argument("--denoiser", default=None, help="custom path: denoiser key")
    p.add_argument("--separator", default=None, help="custom path: separator key")
    p.add_argument("--speakers", type=int, default=None)
    p.add_argument("--count-on", default="original", choices=("original", "denoised"))
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("demo", help="generate a synthetic multi-speaker test file")
    p.add_argument("--speakers", type=int, default=4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.set_defaults(func=cmd_demo)

    return ap


BANNER = """
  Denoise & Separate
  ------------------
  No command given, so starting the web UI (that is what you usually want).

  Other commands:
    python run.py doctor                  what is installed, what is missing
    python run.py demo                    make a synthetic 4-speaker test file
    python run.py methods | paths         list backends / pipeline presets
    python run.py denoise  IN.wav  --method deepfilternet -o clean.wav
    python run.py separate IN.wav  --method pyannote --speakers 4 -o out/
    python run.py pipeline IN.wav  --paths path1,path4
    python run.py serve --port 8080       run the UI on another port

  In PyCharm: Run > Edit Configurations... > Parameters, to pass a command.
"""


def main(argv: "list[str] | None" = None) -> int:
    ensure_dirs()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # bare `python run.py` (e.g. PyCharm's green Run button) -> serve
        print(BANNER, flush=True)
        argv = ["serve"]
    args = build_parser().parse_args(argv)

    # Same contract the server honours: a failure is a sentence plus a fix, and
    # the traceback goes to data/cache/errors.log rather than the user's screen.
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130
    except Exception as exc:
        from src.core.errors import error_summary, log_exception

        log_exception("cli:%s" % argv[0], exc)
        print("\n! %s" % error_summary(exc, context=argv[0]), file=sys.stderr)
        print("  full traceback: data/cache/errors.log", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
