"""Measure complete local pipelines against recorded speaker references.

    python scripts/benchmark_paths.py --paths path1,path3,path4,path5,path6,path8,path9

Cloud calls are excluded. Gated diarization is opt-in (--include-gated), since
installed packages and a token do not guarantee access to the model weights.
Includes separation-only and gentle-DSP controls for overlapping speech.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_audio import speaker_cases, score_sources
from src.core.audio_io import load_audio, save_audio
from src.core.types import AudioBuffer
from src.core.utils import write_json, validate_component
from src.pipeline.paths import PipelinePath, PRESET_PATHS, availability
from src.pipeline.runner import PipelineRunner, PipelineOptions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data/datasets/_download/LibriSpeech/dev-clean')
    ap.add_argument('--paths', default='path1,path3,path4,path5,path6,path8,path9')
    ap.add_argument('--seconds', type=float, default=4)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--label', default='verified')
    ap.add_argument('--include-gated', action='store_true')
    args = ap.parse_args()
    validate_component(args.label)
    folder = ROOT / 'data/outputs' / f'path_audit_{args.label}'
    folder.mkdir(parents=True, exist_ok=True)
    selected = set(args.paths.split(','))
    paths, rows = [], []
    for path in PRESET_PATHS:
        if path.id not in selected:
            continue
        ready = availability(path)
        if path.denoiser == 'api_huggingface' or path.separator == 'api_huggingface':
            rows.append(dict(path=path.id, status='adapter_tests_only', reason='cloud; no recording uploaded'))
        elif ready['separator_effective'] == 'pyannote' and not args.include_gated:
            rows.append(dict(path=path.id, status='gated_model_not_run', reason='use --include-gated after accepting model conditions'))
        elif not ready['available']:
            rows.append(dict(path=path.id, status='unavailable', reason=ready))
        else:
            paths.append(path)
    paths += [PipelinePath('control_asteroid', 'Separation only', 'none', 'convtasnet_asteroid'),
              PipelinePath('mmse_asteroid', 'Gentle DSP plus Asteroid', 'wiener_mmse', 'convtasnet_asteroid')]
    for case, clean, refs in speaker_cases(args.corpus, args.seconds, args.seed):
        audio = clean
        if case == 'overlap':
            noise = np.random.default_rng(args.seed).normal(size=clean.n_samples).astype(np.float32)
            noise *= np.sqrt(np.mean(clean.samples**2) / (np.mean(noise**2) * 10**.5))
            audio = AudioBuffer(clean.samples + noise, clean.sr)
        input_path = folder / f'{case}.wav'
        save_audio(input_path, audio)
        runner = PipelineRunner(job_id=case, out_root=folder,
                                progress=lambda event: print(event.get('message', ''), flush=True))
        report = runner.run(input_path, paths, PipelineOptions(num_speakers=2, save_waveforms=False))
        for result in report['paths']:
            row = dict(case=case, path=result['id'], status=result['status'],
                       denoiser=result['denoiser'], separator=result['separator'])
            if result['status'] == 'ok':
                from types import SimpleNamespace
                tracks = [SimpleNamespace(audio=load_audio(track['path'])) for track in result['tracks']]
                row.update(score_sources(tracks, refs, audio))
            else:
                row['error'] = result.get('error')
            rows.append(row)
            print(row, flush=True)
        write_json(folder / 'report.json', dict(rows=rows))
    print(f'Wrote {folder / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
