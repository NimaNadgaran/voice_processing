"""Measure how good the paths actually are, against known ground truth.

Unlike the reference-free scores the app shows for an arbitrary upload, this
harness *builds* its test set, so it knows exactly who said what and can report
real accuracy:

    python scripts/benchmark.py --source synthetic --cases 6
    python scripts/benchmark.py --source librispeech --corpus data/datasets/_download/LibriSpeech/dev-clean
    python scripts/benchmark.py --paths path1,path4,path8 --speakers 2,3,4 --snr 5,15

What is measured
----------------
Speaker counting
    ``count_exact``  -- did it get the number of speakers exactly right?
    ``count_error``  -- signed error (predicted - true), averaged

Separation (each output track is matched to the ground-truth speaker it
overlaps most, then)
    ``coverage``     -- fraction of true speakers that got their own track
    ``purity``       -- of the energy in a track, how much belongs to the
                        speaker it was assigned (1.0 = perfectly clean)
    ``si_sdr_in/out``-- scale-invariant SDR against the true speaker signal,
                        before (mixture) and after separation
    ``si_sdri``      -- the improvement, which is THE separation metric

Denoising
    ``snr_gain``     -- true SNR improvement, computed against the clean mixture
    ``si_sdr_gain``  -- SI-SDR of the denoised signal vs the clean mixture

Everything is written to ``data/outputs/benchmark_<stamp>.json`` alongside a
printed table.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.audio_io import load_audio, save_audio  # noqa: E402
from src.core.metrics import si_sdr  # noqa: E402
from src.core.types import AudioBuffer  # noqa: E402
from src.core.utils import OUT_DIR, human_time, write_json  # noqa: E402
from src.denoising import denoise as run_denoise  # noqa: E402
from src.pipeline import PRESETS_BY_ID, resolve_path  # noqa: E402
from src.separation import count_speakers, separate as run_separate  # noqa: E402

EPS = 1e-12


# --------------------------------------------------------------------------- #
#  building test cases with known ground truth
# --------------------------------------------------------------------------- #
def synthetic_case(n_speakers: int, snr_db: float, seconds: float, sr: int, seed: int):
    """Reuse the demo generator: formant voices, turn taking, realistic noise."""
    from make_demo_audio import build_conversation  # same folder

    noisy, tracks, clean = build_conversation(n_speakers, seconds, sr, snr_db, seed)
    return (
        AudioBuffer(noisy, sr),
        AudioBuffer(clean, sr),
        [AudioBuffer(t, sr) for t in tracks],
    )


def librispeech_case(
    corpus: Path, n_speakers: int, snr_db: float, seconds: float, sr: int, seed: int
):
    """Real recorded speech: pick N different speakers and lay out turns."""
    rng = np.random.default_rng(seed)
    speakers = sorted(p for p in corpus.iterdir() if p.is_dir())
    if len(speakers) < n_speakers:
        raise SystemExit("only %d speaker folders in %s" % (len(speakers), corpus))

    chosen = [speakers[i] for i in rng.choice(len(speakers), n_speakers, replace=False)]
    n = int(seconds * sr)
    tracks = [np.zeros(n, dtype=np.float32) for _ in range(n_speakers)]

    for idx, speaker_dir in enumerate(chosen):
        files = sorted(speaker_dir.rglob("*.flac")) or sorted(speaker_dir.rglob("*.wav"))
        if not files:
            raise SystemExit("no audio under %s" % speaker_dir)
        # concatenate a few utterances so we have enough material
        pieces = []
        total = 0
        for f in rng.permutation(np.array(files, dtype=object)):
            buf = load_audio(Path(str(f)), target_sr=sr)
            pieces.append(buf.samples)
            total += buf.n_samples
            if total >= n:
                break
        voice = np.concatenate(pieces)[:n]
        if len(voice) < n:
            voice = np.pad(voice, (0, n - len(voice)))
        rms = float(np.sqrt(np.mean(voice**2))) + EPS
        tracks[idx] = (voice / rms * 0.05).astype(np.float32)

    # turn taking: 2-5 s turns, round robin, occasional short overlap
    gates = [np.zeros(n, dtype=np.float32) for _ in range(n_speakers)]
    fade = int(0.05 * sr)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
    pos, who = 0, 0
    while pos < n:
        dur = int(rng.uniform(2.0, 5.0) * sr)
        end = min(pos + dur, n)
        g = gates[who % n_speakers]
        g[pos:end] = 1.0
        if end - pos > 2 * fade:
            g[pos:pos + fade] = ramp
            g[end - fade:end] = ramp[::-1]
        overlap = int(rng.uniform(0.0, 0.6) * sr) if rng.random() < 0.3 else 0
        pos = max(pos + 1, end - overlap)
        who += 1

    tracks = [t * g for t, g in zip(tracks, gates)]
    clean = np.sum(tracks, axis=0)
    peak = float(np.max(np.abs(clean))) or 1.0
    clean = (clean / peak * 0.7).astype(np.float32)
    tracks = [(t / peak * 0.7).astype(np.float32) for t in tracks]

    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.convolve(noise, np.ones(8, dtype=np.float32) / 8, mode="same")
    noise += 0.3 * np.sin(2 * np.pi * 50.0 * np.arange(n) / sr).astype(np.float32)
    sig_p = float(np.mean(clean**2)) + EPS
    noise *= np.sqrt(sig_p / ((float(np.mean(noise**2)) + EPS) * 10 ** (snr_db / 10.0)))

    noisy = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)
    return AudioBuffer(noisy, sr), AudioBuffer(clean, sr), [AudioBuffer(t, sr) for t in tracks]


# --------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------- #
def match_tracks(estimates: Sequence[AudioBuffer], truth: Sequence[AudioBuffer]) -> Dict[str, Any]:
    """Assign each estimated track to the truth speaker it best explains."""
    if not estimates:
        return {"coverage": 0.0, "purity": 0.0, "si_sdr_out": None, "assignment": []}

    n = min([e.n_samples for e in estimates] + [t.n_samples for t in truth])
    est = np.stack([e.samples[:n] for e in estimates])
    ref = np.stack([t.samples[:n] for t in truth])

    # energy of each truth speaker inside each estimated track's active region
    assignment, purities, sdrs = [], [], []
    for i in range(est.shape[0]):
        active = np.abs(est[i]) > (0.02 * (np.max(np.abs(est[i])) + EPS))
        if active.sum() < 10:
            assignment.append(-1)
            purities.append(0.0)
            continue
        energies = np.array([float(np.mean(ref[j][active] ** 2)) for j in range(ref.shape[0])])
        best = int(np.argmax(energies))
        assignment.append(best)
        purities.append(float(energies[best] / (energies.sum() + EPS)))
        sdrs.append(si_sdr(est[i], ref[best]))

    covered = {a for a in assignment if a >= 0}
    return {
        "coverage": round(len(covered) / max(len(truth), 1), 4),
        "purity": round(float(np.mean(purities)) if purities else 0.0, 4),
        "si_sdr_out": round(float(np.mean(sdrs)), 3) if sdrs else None,
        "assignment": assignment,
    }


def mixture_baseline_si_sdr(mixture: AudioBuffer, truth: Sequence[AudioBuffer]) -> float:
    """SI-SDR of the unseparated mixture against each speaker -- the 'do nothing' score."""
    n = min([mixture.n_samples] + [t.n_samples for t in truth])
    return float(np.mean([si_sdr(mixture.samples[:n], t.samples[:n]) for t in truth]))


# --------------------------------------------------------------------------- #
def run_case(case: Dict[str, Any], path_ids: List[str], pin_speakers: bool) -> List[Dict[str, Any]]:
    noisy, clean, truth = case["noisy"], case["clean"], case["truth"]
    rows: List[Dict[str, Any]] = []

    est = count_speakers(noisy)
    baseline = mixture_baseline_si_sdr(noisy, truth)

    for path_id in path_ids:
        path = resolve_path({"id": path_id}) if path_id in PRESETS_BY_ID else resolve_path(
            {"denoiser": path_id.split("+")[0], "separator": path_id.split("+")[1]}
        )
        started = time.perf_counter()

        den = run_denoise(noisy, method=path.denoiser)
        if den.metrics.get("error"):
            rows.append({"case": case["name"], "path": path_id, "error": den.metrics["error"]})
            continue

        n_hint = len(truth) if pin_speakers else None
        sep = run_separate(den.audio, method=path.separator, num_speakers=n_hint)
        elapsed = time.perf_counter() - started
        if sep.metrics.get("error"):
            rows.append({"case": case["name"], "path": path_id, "error": sep.metrics["error"]})
            continue

        scored = match_tracks([t.audio for t in sep.tracks], truth)
        rows.append({
            "case": case["name"],
            "n_true": len(truth),
            "snr_db": case["snr_db"],
            "path": path_id,
            "denoiser": path.denoiser,
            "separator": path.separator,
            # -- denoising, against the true clean mixture --
            "denoise_si_sdr_in": round(si_sdr(noisy.samples, clean.samples), 3),
            "denoise_si_sdr_out": round(si_sdr(den.audio.samples, clean.samples), 3),
            "denoise_si_sdr_gain": round(
                si_sdr(den.audio.samples, clean.samples) - si_sdr(noisy.samples, clean.samples), 3),
            # -- counting --
            "n_estimated": int(est["n_speakers"]),
            "count_exact": bool(int(est["n_speakers"]) == len(truth)),
            "count_error": int(est["n_speakers"]) - len(truth),
            "count_confidence": round(float(est["confidence"]), 3),
            # -- separation --
            "n_output": len(sep.tracks),
            "coverage": scored["coverage"],
            "purity": scored["purity"],
            "si_sdr_baseline": round(baseline, 3),
            "si_sdr_out": scored["si_sdr_out"],
            "si_sdri": (round(scored["si_sdr_out"] - baseline, 3)
                        if scored["si_sdr_out"] is not None else None),
            "seconds": round(elapsed, 2),
            "realtime_factor": round(elapsed / max(noisy.duration, EPS), 3),
        })
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("synthetic", "librispeech"), default="synthetic")
    ap.add_argument("--corpus", default="data/datasets/_download/LibriSpeech/dev-clean")
    ap.add_argument("--paths", default="", help="comma separated (default: every ready preset)")
    ap.add_argument("--speakers", default="2,3,4", help="speaker counts to test")
    ap.add_argument("--snr", default="5,15", help="mixture SNRs in dB")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pin-speakers", action="store_true",
                    help="tell the separator the true count (isolates separation from counting)")
    ap.add_argument("--save-audio", action="store_true", help="also write the test mixtures")
    args = ap.parse_args()

    if args.paths:
        path_ids = [p.strip() for p in args.paths.split(",") if p.strip()]
    else:
        path_ids = [pid for pid, p in PRESETS_BY_ID.items() if p.to_dict()["available"]]
    if not path_ids:
        print("no ready paths -- install a backend or pass --paths")
        return 1

    speaker_counts = [int(x) for x in args.speakers.split(",") if x.strip()]
    snrs = [float(x) for x in args.snr.split(",") if x.strip()]

    print("=" * 78)
    print(" benchmark: %s | paths: %s" % (args.source, ", ".join(path_ids)))
    print(" %d speaker counts x %d SNRs = %d cases, %.0fs each%s"
          % (len(speaker_counts), len(snrs), len(speaker_counts) * len(snrs), args.seconds,
             "  (speaker count pinned)" if args.pin_speakers else ""))
    print("=" * 78)

    corpus = ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    if args.source == "librispeech" and not corpus.exists():
        print("corpus not found: %s" % corpus)
        return 1

    rows: List[Dict[str, Any]] = []
    for seed_i, (n_spk, snr) in enumerate(itertools.product(speaker_counts, snrs)):
        name = "%dspk_%.0fdB" % (n_spk, snr)
        print("\n[case] %s" % name, flush=True)
        if args.source == "synthetic":
            noisy, clean, truth = synthetic_case(n_spk, snr, args.seconds, args.sr, args.seed + seed_i)
        else:
            noisy, clean, truth = librispeech_case(
                corpus, n_spk, snr, args.seconds, args.sr, args.seed + seed_i)

        if args.save_audio:
            out = OUT_DIR / "benchmark_cases"
            out.mkdir(parents=True, exist_ok=True)
            save_audio(out / ("%s_mixture.wav" % name), noisy)

        case = {"name": name, "noisy": noisy, "clean": clean, "truth": truth, "snr_db": snr}
        case_rows = run_case(case, path_ids, args.pin_speakers)
        for r in case_rows:
            if r.get("error"):
                print("   %-10s FAILED: %s" % (r["path"], r["error"][:60]))
            else:
                print("   %-10s count %d/%d  coverage %.2f  purity %.2f  SI-SDRi %+.1f dB  (%.1fs)"
                      % (r["path"], r["n_estimated"], r["n_true"], r["coverage"], r["purity"],
                         r["si_sdri"] if r["si_sdri"] is not None else float("nan"), r["seconds"]))
        rows.extend(case_rows)

    # ---------------- summary per path ---------------------------------- #
    print("\n" + "=" * 78)
    print(" SUMMARY (mean over %d cases)" % (len(speaker_counts) * len(snrs)))
    print("=" * 78)
    header = "%-10s %8s %8s %8s %9s %9s %8s" % (
        "path", "count%", "coverage", "purity", "SI-SDRi", "denoise", "xRT")
    print(header)
    print("-" * len(header))

    summary = {}
    for path_id in path_ids:
        got = [r for r in rows if r["path"] == path_id and not r.get("error")]
        if not got:
            print("%-10s  (all cases failed)" % path_id)
            continue
        agg = {
            "count_accuracy": round(100.0 * float(np.mean([r["count_exact"] for r in got])), 1),
            "coverage": round(float(np.mean([r["coverage"] for r in got])), 3),
            "purity": round(float(np.mean([r["purity"] for r in got])), 3),
            "si_sdri": round(float(np.mean([r["si_sdri"] for r in got if r["si_sdri"] is not None])), 2),
            "denoise_si_sdr_gain": round(float(np.mean([r["denoise_si_sdr_gain"] for r in got])), 2),
            "realtime_factor": round(float(np.mean([r["realtime_factor"] for r in got])), 3),
            "cases": len(got),
        }
        summary[path_id] = agg
        print("%-10s %7.0f%% %8.2f %8.2f %+8.2f %+8.2f %8.3f"
              % (path_id, agg["count_accuracy"], agg["coverage"], agg["purity"],
                 agg["si_sdri"], agg["denoise_si_sdr_gain"], agg["realtime_factor"]))

    print("\n count%%   = how often the speaker count was exactly right")
    print(" coverage = fraction of real speakers that got their own file (1.0 is perfect)")
    print(" purity   = how much of a track's energy really is the assigned speaker")
    print(" SI-SDRi  = separation improvement over doing nothing, in dB (higher is better)")
    print(" denoise  = SI-SDR gain of the denoised signal against the true clean mixture")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = OUT_DIR / ("benchmark_%s.json" % stamp)
    write_json(report_path, {
        "source": args.source,
        "paths": path_ids,
        "speakers": speaker_counts,
        "snrs": snrs,
        "seconds": args.seconds,
        "pinned": args.pin_speakers,
        "rows": rows,
        "summary": summary,
    })
    print("\n wrote %s" % report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
