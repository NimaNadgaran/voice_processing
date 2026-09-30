"""Generate a synthetic multi-speaker conversation for testing the pipeline.

No dataset download needed: each "speaker" is a simple source-filter voice
(glottal pulse train at their own pitch, pushed through their own formant
resonators) so the voices genuinely differ in pitch and timbre -- enough for the
clustering separator to have something real to work with.

    python scripts/make_demo_audio.py --speakers 4 --seconds 30 --snr 8

Writes ``data/raw/demo_conversation.wav`` plus the per-speaker ground-truth
tracks in ``data/raw/demo_truth/`` so you can score the separators for real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.audio_io import save_audio  # noqa: E402
from src.core.utils import RAW_DIR  # noqa: E402

VOICES = [
    # (f0 Hz, formants Hz, jitter, brightness)
    (112.0, (620.0, 1180.0, 2500.0), 0.020, 0.7),
    (196.0, (740.0, 1620.0, 2800.0), 0.028, 1.0),
    (145.0, (500.0, 1500.0, 2350.0), 0.015, 0.85),
    (232.0, (820.0, 1900.0, 3100.0), 0.032, 1.15),
    (98.0, (560.0, 1000.0, 2200.0), 0.018, 0.6),
    (168.0, (680.0, 1350.0, 2650.0), 0.024, 0.95),
]


def synth_voice(seconds: float, sr: int, voice, rng: np.random.Generator) -> np.ndarray:
    """Glottal pulse train -> formant filter -> syllable envelope."""
    f0, formants, jitter, brightness = voice
    n = int(seconds * sr)
    t = np.arange(n) / sr

    # pitch contour: slow drift + micro jitter (this is what makes it sound alive)
    drift = 1.0 + 0.06 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28))
    micro = 1.0 + jitter * rng.standard_normal(n).cumsum() / max(np.sqrt(n), 1)
    phase = 2 * np.pi * np.cumsum(f0 * drift * micro) / sr

    # band-limited pulse train (a few harmonics, 1/k rolloff)
    src = np.zeros(n, dtype=np.float32)
    n_harm = int(min(24, (sr / 2) / f0))
    for k in range(1, n_harm + 1):
        src += (brightness / k) * np.sin(k * phase).astype(np.float32)
    src += 0.05 * rng.standard_normal(n).astype(np.float32)  # breathiness

    # formant resonators (2-pole IIR per formant)
    out = np.zeros(n, dtype=np.float32)
    for f_c, gain in zip(formants, (1.0, 0.6, 0.35)):
        bw = 90.0 + 0.1 * f_c
        r = float(np.exp(-np.pi * bw / sr))
        theta = 2 * np.pi * f_c / sr
        a1, a2 = -2 * r * np.cos(theta), r * r
        y = np.zeros(n, dtype=np.float32)
        y1 = y2 = 0.0
        for i in range(n):
            y[i] = src[i] - a1 * y1 - a2 * y2
            y2, y1 = y1, y[i]
        out += gain * y
    out /= max(float(np.max(np.abs(out))), 1e-6)

    # syllable envelope ~4.5 Hz + pauses between words
    syll = 0.5 * (1 + np.sin(2 * np.pi * 4.5 * t + rng.uniform(0, 6.28)))
    syll = np.clip(syll**1.5, 0.05, 1.0)
    return (out * syll).astype(np.float32)


def build_conversation(n_speakers: int, seconds: float, sr: int, snr_db: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = [synth_voice(seconds, sr, VOICES[i % len(VOICES)], rng) for i in range(n_speakers)]

    # turn taking: 2-5 s turns, round robin with a little overlap
    n = int(seconds * sr)
    gates = [np.zeros(n, dtype=np.float32) for _ in range(n_speakers)]
    pos, speaker = 0, 0
    fade = int(0.05 * sr)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
    while pos < n:
        dur = int(rng.uniform(2.0, 5.0) * sr)
        end = min(pos + dur, n)
        g = gates[speaker % n_speakers]
        g[pos:end] = 1.0
        if end - pos > 2 * fade:
            g[pos:pos + fade] = ramp
            g[end - fade:end] = ramp[::-1]
        overlap = int(rng.uniform(0.0, 0.6) * sr) if rng.random() < 0.35 else 0
        pos = max(pos + 1, end - overlap)
        speaker += 1

    tracks = [t * g for t, g in zip(truth, gates)]
    mix = np.sum(tracks, axis=0)
    mix /= max(float(np.max(np.abs(mix))), 1e-6)
    mix *= 0.7

    # noise: pink-ish hiss + 50 Hz hum + occasional keyboard clicks
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.convolve(noise, np.ones(8, dtype=np.float32) / 8, mode="same")
    noise += 0.35 * np.sin(2 * np.pi * 50.0 * np.arange(n) / sr).astype(np.float32)
    for _ in range(int(seconds / 3)):
        at = int(rng.uniform(0, n - 400))
        noise[at:at + 400] += rng.standard_normal(400).astype(np.float32) * 2.0

    sig_p = float(np.mean(mix**2)) + 1e-12
    noise_p = float(np.mean(noise**2)) + 1e-12
    noise *= np.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))

    noisy = np.clip(mix + noise, -1.0, 1.0).astype(np.float32)
    return noisy, tracks, mix.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speakers", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--snr", type=float, default=8.0, help="mixture SNR in dB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(RAW_DIR / "demo_conversation.wav"))
    args = ap.parse_args()

    noisy, tracks, clean = build_conversation(
        args.speakers, args.seconds, args.sr, args.snr, args.seed
    )

    out = Path(args.out)
    save_audio(out, noisy, args.sr)
    save_audio(out.with_name(out.stem + "_clean.wav"), clean, args.sr)

    truth_dir = out.parent / (out.stem + "_truth")
    truth_dir.mkdir(parents=True, exist_ok=True)
    for i, track in enumerate(tracks):
        save_audio(truth_dir / ("speaker_%02d.wav" % (i + 1)), track, args.sr)

    print("wrote %s  (%.1fs, %d speakers, %.1f dB SNR)" % (out, args.seconds, args.speakers, args.snr))
    print("ground truth ->", truth_dir)


if __name__ == "__main__":
    main()
