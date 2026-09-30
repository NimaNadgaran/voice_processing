"""Run every local audio backend and record reference-based quality and readiness.

    python scripts/audit_audio.py
    python scripts/audit_audio.py --methods sepformer,convtasnet_asteroid --seconds 4

Uses two recorded speakers from LibriSpeech dev-clean for overlapping speech,
turn-taking, and added-noise checks. Cloud calls are excluded: their request and
response adapters are tested in tests/test_audio_backends.py without uploading
recordings. The report distinguishes successful runs, failures and missing
dependencies. A success means valid audio, not perfect separation quality.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.audio_io import save_audio
from src.core.metrics import si_sdr
from src.core.registry import list_denoisers, list_separators, get_denoiser, get_separator
from src.core.types import AudioBuffer
from src.core.utils import write_json
from scripts.benchmark_denoising import recorded_cases


def speaker_cases(corpus, seconds=4, seed=99, speakers=2):
    clips = list(recorded_cases(corpus, speakers, seconds, (16000,), (5,), seed))
    refs = [clips[3 * i][2].samples for i in range(speakers)]
    length = min(map(len, refs))
    refs = [x[:length] for x in refs]
    yield 'overlap', AudioBuffer(sum(refs), 16000), refs
    turns = [np.concatenate([refs[i] if i == j else np.zeros(length)
                             for j in range(speakers)]) for i in range(speakers)]
    yield 'turn_taking', AudioBuffer(sum(turns), 16000), turns


def score_sources(tracks, refs, mixture):
    if len(tracks) != len(refs):
        return dict(output_count=len(tracks), reference_count=len(refs), si_sdri_db=None)
    scores = []
    for perm in itertools.permutations(range(len(refs))):
        scores.append(np.mean([si_sdr(tracks[i].audio.samples, refs[j]) -
                               si_sdr(mixture.samples, refs[j]) for i, j in enumerate(perm)]))
    return dict(output_count=len(tracks), reference_count=len(refs), si_sdri_db=float(max(scores)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data/datasets/_download/LibriSpeech/dev-clean')
    ap.add_argument('--seconds', type=float, default=4)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--speakers', type=int, choices=(2, 3), default=2)
    ap.add_argument('--methods', default='', help='optional comma-separated method filter')
    ap.add_argument('--save-audio', action='store_true')
    ap.add_argument('--label', default='all_backends')
    args = ap.parse_args()
    from src.core.utils import validate_component
    validate_component(args.label)
    folder = ROOT / 'data/outputs' / f'audio_audit_{args.label}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    requested = set(args.methods.split(',')) if args.methods else None
    cases = list(speaker_cases(args.corpus, args.seconds, args.seed, args.speakers))
    # Ordinary speech enhancement and overlap preservation are different tasks.
    # Keep both visible: passing a format check is not a quality guarantee.
    denoise_cases = [('single_speaker', AudioBuffer(cases[0][2][0], 16000), []), cases[0]]
    for kind, methods in (('denoise', list_denoisers()), ('separate', list_separators())):
        for info in methods:
            if requested and info.key not in requested:
                continue
            if not info.offline:
                rows.append(dict(kind=kind, method=info.key, status='adapter_tests_only', reason='cloud'))
                continue
            if not info.available:
                rows.append(dict(kind=kind, method=info.key, status='unavailable', reason=info.unavailable_reason))
                print(f'{kind} {info.key}: unavailable ({info.unavailable_reason})', flush=True)
                continue
            backend = get_denoiser(info.key) if kind == 'denoise' else get_separator(info.key)
            for case, clean, refs in cases if kind == 'separate' else denoise_cases:
                audio = clean
                if kind == 'denoise':
                    noise = np.random.default_rng(args.seed).normal(size=clean.n_samples).astype(np.float32)
                    noise *= np.sqrt(np.mean(clean.samples**2) / (np.mean(noise**2) * 10**.5))
                    audio = AudioBuffer(clean.samples + noise, clean.sr)
                print(f'Running {kind}/{info.key}/{case}...', flush=True)
                start = time.perf_counter()
                result = backend.safe_run(audio) if kind == 'denoise' else backend.safe_run(audio, args.speakers)
                row = dict(kind=kind, method=info.key, case=case, seconds=time.perf_counter() - start,
                           metrics=result.metrics, status='failed' if result.metrics.get('error') else 'ok')
                if row['status'] == 'ok':
                    buffers = [result.audio] if kind == 'denoise' else [t.audio for t in result.tracks]
                    if any(b.sr != audio.sr or b.n_samples != audio.n_samples or
                           not np.isfinite(b.samples).all() for b in buffers):
                        row.update(status='failed', reason='Invalid sample rate, length, or non-finite audio')
                    if kind == 'denoise':
                        row['si_sdr_gain_db'] = si_sdr(result.audio.samples, clean.samples) - si_sdr(audio.samples, clean.samples)
                    else:
                        row.update(score_sources(result.tracks, refs, audio))
                    if args.save_audio:
                        save_audio(folder / f'{kind}_{info.key}_{case}_input.wav', audio)
                        for i, buf in enumerate(buffers):
                            save_audio(folder / f'{kind}_{info.key}_{case}_{i + 1}.wav', buf)
                rows.append(row)
                write_json(folder / 'report.json', dict(rows=rows))
                print(f"  {row['status']}: gain {row.get('si_sdri_db', row.get('si_sdr_gain_db'))} dB"
                      + (f" ({result.metrics['error']})" if result.metrics.get('error') else ''), flush=True)
            backend.unload()
    write_json(folder / 'report.json', dict(rows=rows))
    print(f'Wrote {folder / "report.json"}')
    return 1 if any(r['status'] == 'failed' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
