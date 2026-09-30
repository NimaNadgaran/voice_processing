"""Measure denoisers against recorded clean speech with reproducible added noise.

Examples:
    python scripts/benchmark_denoising.py --label before
    python scripts/benchmark_denoising.py --label after --save-audio
    python scripts/benchmark_denoising.py --input recording.mp4 --start 15 --seconds 12 --save-audio

An arbitrary --input has no clean reference: its output is for listening, and
only reference-free metrics are reported. Corpus cases have true SI-SDR gains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.audio_io import load_audio, resample, save_audio
from src.core.metrics import si_sdr
from src.core.types import AudioBuffer
from src.core.utils import write_json
from src.denoising import denoise


def recorded_cases(corpus, clips=3, seconds=6., rates=(16000, 48000), snrs=(5.,), seed=41):
    # Select distinct speakers, then one utterance each; no synthetic vowels.
    speakers = sorted(p for p in Path(corpus).iterdir() if p.is_dir())
    if len(speakers) < clips:
        raise ValueError(f"Need at least {clips} speaker folders in {corpus}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(speakers), clips, replace=False)
    for clip, index in enumerate(selected):
        files = sorted(speakers[index].rglob("*.flac")) or sorted(speakers[index].rglob("*.wav"))
        if not files:
            raise ValueError(f"No audio in {speakers[index]}")
        speech = load_audio(files[int(rng.integers(len(files)))], target_sr=16000)
        speech.samples = speech.samples[:int(seconds * speech.sr)]
        speech.samples *= .06 / max(speech.rms(), 1e-8)
        if speech.peak() > .7:
            speech.samples *= .7 / speech.peak()
        for sr in rates:
            voice = resample(speech, sr).samples
            clean = np.pad(voice, (sr // 2, sr // 2))
            for kind in ("white", "hum", "changing"):
                noise_rng = np.random.default_rng(seed + clip * 7)
                noise = noise_rng.standard_normal(len(clean)).astype(np.float32)
                t = np.arange(len(clean)) / sr
                if kind == "hum":
                    noise *= .25
                    noise += np.sin(2 * np.pi * 60 * t) + .4 * np.sin(2 * np.pi * 120 * t)
                elif kind == "changing":
                    noise *= .25 + 1.5 * (t > t[-1] / 2)
                for snr in snrs:
                    scaled = noise * np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10**(snr / 10)))
                    name = f"speaker{speakers[index].name}_{kind}_{sr}_{snr:g}dB"
                    yield name, AudioBuffer(clean + scaled, sr), AudioBuffer(clean, sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/datasets/_download/LibriSpeech/dev-clean")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--rates", default="16000,48000")
    ap.add_argument("--snrs", default="5")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--methods", default="spectral_gate,wiener_mmse,rnnoise,demucs_denoiser,local_unet")
    ap.add_argument("--label", default="denoising")
    ap.add_argument("--save-audio", action="store_true")
    args = ap.parse_args()
    if args.seconds <= 0 or args.start < 0:
        ap.error('--seconds must be positive and --start nonnegative')
    from src.core.utils import validate_component
    validate_component(args.label)
    folder = ROOT / "data/outputs" / f"denoise_tuning_{args.label}"
    folder.mkdir(parents=True, exist_ok=True)
    if args.input:
        audio = load_audio(args.input)
        start = int(args.start * audio.sr)
        audio = AudioBuffer(audio.samples[start:start + int(args.seconds * audio.sr)], audio.sr)
        if not audio.n_samples:
            ap.error('selected recording interval contains no audio')
        cases = [("recording", audio, None)]
    else:
        cases = recorded_cases(args.corpus, args.clips, args.seconds,
                               tuple(map(int, args.rates.split(','))),
                               tuple(map(float, args.snrs.split(','))), args.seed)
    methods = args.methods.split(',')
    rows = []
    for name, noisy, clean in cases:
        print(name, flush=True)
        if args.save_audio:
            save_audio(folder / f"{name}_input.wav", noisy)
            if clean is not None:
                save_audio(folder / f"{name}_reference.wav", clean)
        for method in methods:
            result = denoise(noisy, method)
            if result.audio.sr != noisy.sr or result.audio.n_samples != noisy.n_samples or not np.isfinite(result.audio.samples).all():
                result.metrics['error'] = 'Invalid output sample rate, length, or non-finite audio'
            row = dict(case=name, method=method, seconds=result.elapsed, metrics=result.metrics)
            if not result.metrics.get("error") and clean is not None:
                row["si_sdr_gain_db"] = si_sdr(result.audio.samples, clean.samples) - si_sdr(noisy.samples, clean.samples)
                row["snr_gain_db"] = 10 * np.log10(np.sum((noisy.samples - clean.samples)**2) /
                                                    max(np.sum((result.audio.samples - clean.samples)**2), 1e-12))
            rows.append(row)
            if args.save_audio and not result.metrics.get("error"):
                save_audio(folder / f"{name}_{method}.wav", result.audio)
            gain = f"{row['si_sdr_gain_db']:+6.2f} dB" if 'si_sdr_gain_db' in row else 'no clean reference'
            print(f"  {method:18s} {gain}  {result.elapsed:.2f}s"
                  + (f"  ERROR: {result.metrics['error']}" if result.metrics.get('error') else ''), flush=True)
        # Keep partial results if a heavy model or later case fails.
        write_json(folder / "report.json", {"options": vars(args), "rows": rows})
    summary = {}
    for method in methods:
        gains = [r['si_sdr_gain_db'] for r in rows if r['method'] == method and 'si_sdr_gain_db' in r]
        if gains:
            summary[method] = dict(mean_gain_db=float(np.mean(gains)), worst_gain_db=float(min(gains)), cases=len(gains))
    write_json(folder / "report.json", {"options": vars(args), "summary": summary, "rows": rows})
    for method, score in summary.items():
        print(f"{method:18s} mean {score['mean_gain_db']:+.2f} dB, worst {score['worst_gain_db']:+.2f} dB")
    print(f"Wrote {folder / 'report.json'}")
    return 1 if any(row['metrics'].get('error') for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
