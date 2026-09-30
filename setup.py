"""One-command local model training after cloning this repository.

    python setup.py
    python setup.py --models local_unet --estimate-only
    python setup.py --models local_unet,local_convtasnet --preset base

This is a training launcher, not a setuptools packaging script. With no flags
it prepares a venv and a speech corpus, measures both models, prints the queue's
finish times, and trains the denoiser followed by the separator. Download time
is reported separately; ETAs are estimates measured on the current computer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import venv
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_MODELS = ('local_unet', 'local_convtasnet')
TRAINERS = {'local_unet': 'src.denoising.training.train_unet',
            'local_convtasnet': 'src.separation.training.train_convtasnet'}
ALIASES = {'denoise': 'local_unet', 'denoise_unet': 'local_unet',
           'separate': 'local_convtasnet', 'convtasnet': 'local_convtasnet'}
CORPUS_URL = 'https://www.openslr.org/resources/12/train-clean-100.tar.gz'


def select_models(models=None):
    if models is None or models == 'all':
        return list(LOCAL_MODELS)
    if isinstance(models, str):
        models = [s.strip() for s in models.split(',') if s.strip()]
    selected = []
    for model in models:
        model = ALIASES.get(model, model)
        if model not in TRAINERS:
            raise ValueError(f"Unknown local model {model!r}; choose {', '.join(LOCAL_MODELS)}")
        if model not in selected:
            selected.append(model)
    if not selected:
        raise ValueError('Select at least one local model')
    return selected


def _run(command, capture=False):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', str(min(4, os.cpu_count() or 1)))
    env.setdefault('MKL_NUM_THREADS', env['OMP_NUM_THREADS'])
    env.setdefault('DS_THREADS', env['OMP_NUM_THREADS'])
    return subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True,
                          text=True, capture_output=capture)


def prepare_environment(install=True):
    python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print(f'Creating project environment with Python {sys.version.split()[0]}...', flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    probe = _run([python, '-c', "import importlib.util; print(all(importlib.util.find_spec(m) is not None for m in ('torch','numpy','scipy','soundfile','yaml','fastapi','uvicorn','multipart')))"], capture=True)
    if probe.stdout.strip() != 'True':
        if not install:
            raise RuntimeError('Required packages are missing; run setup.py without --no-install')
        print('Installing the application and training dependencies...', flush=True)
        _run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt',
              '-r', ROOT / 'requirements-train.txt'])
    return python


def _has_audio(folder):
    folder = Path(folder)
    return folder.is_dir() and any(next(folder.rglob(f'*{ext}'), None) for ext in ('.wav', '.flac'))


def download_corpus():
    folder = ROOT / 'data/datasets/_download'
    corpus = folder / 'LibriSpeech/train-clean-100'
    marker = corpus / '.setup_complete'
    if marker.exists():
        return corpus
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'train-clean-100.tar.gz'
    partial = archive.with_suffix('.gz.part')
    if not archive.exists():
        print('Downloading LibriSpeech train-clean-100 (~6.4 GB, CC BY 4.0).', flush=True)
        if shutil.disk_usage(folder).free < 15 * 1024**3:
            raise RuntimeError('Need about 15 GB of free disk for the archive and extracted speech')
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(CORPUS_URL, headers={'Range': f'bytes={offset}-'} if offset else {})
        start, last, downloaded = time.monotonic(), 0., 0
        with urllib.request.urlopen(request, timeout=90) as response:
            append = offset > 0 and response.status == 206
            if not append:
                offset = 0
            total = int(response.headers.get('Content-Length', 0)) + offset
            with partial.open('ab' if append else 'wb') as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    elapsed = time.monotonic() - start
                    if elapsed - last >= 5:
                        rate = downloaded / max(elapsed, .001)
                        remaining = max(0, total - offset - downloaded) / max(rate, 1)
                        print(f'Download {(offset + downloaded) / 1024**3:.2f}/{total / 1024**3:.2f} GB; '
                              f'{rate / 1024**2:.1f} MB/s; about {remaining / 60:.1f} minutes left', flush=True)
                        last = elapsed
        if total and offset + downloaded != total:
            raise RuntimeError('Download incomplete. Run setup.py again to resume it.')
        partial.replace(archive)
    print('Extracting training speech...', flush=True)
    # The data filter rejects escaping paths, links, and special files. Validate
    # explicitly too, for Python versions older than the extraction filter.
    with tarfile.open(archive) as tar:
        target = folder.resolve()
        members = tar.getmembers()
        for member in members:
            path = (target / member.name).resolve()
            if not path.is_relative_to(target) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe corpus archive entry: {member.name}')
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(folder, members=members, filter='data')
        else:
            tar.extractall(folder, members=members)
    if not _has_audio(corpus):
        raise RuntimeError('Downloaded archive contained no training speech')
    marker.write_text('LibriSpeech train-clean-100; https://www.openslr.org/12\n', encoding='utf-8')
    return corpus


def prepare_data(models, clean_dir=None, speakers_dir=None, download=True):
    clean = Path(clean_dir).resolve() if clean_dir else ROOT / 'data/datasets/clean'
    speakers = Path(speakers_dir).resolve() if speakers_dir else ROOT / 'data/datasets/speakers'
    if 'local_unet' in models and clean_dir and not _has_audio(clean):
        raise FileNotFoundError(f'No speech in supplied --clean-dir {clean}')
    if 'local_convtasnet' in models and speakers_dir and not _has_audio(speakers):
        raise FileNotFoundError(f'No speech in supplied --speakers-dir {speakers}')
    if 'local_convtasnet' in models and not speakers_dir and clean_dir and _has_audio(clean):
        # One corpus can serve both models without duplicating recordings.
        speakers = clean
    needed = (('local_unet' in models and not _has_audio(clean)) or
              ('local_convtasnet' in models and not _has_audio(speakers)))
    if needed:
        # Reuse an existing training split, never silently train on dev/test.
        corpus = ROOT / 'data/datasets/_download/LibriSpeech/train-clean-100'
        if not _has_audio(corpus):
            if not download:
                raise FileNotFoundError('Training speech is missing; provide --clean-dir/--speakers-dir or allow the download')
            corpus = download_corpus()
        if not _has_audio(clean):
            if clean_dir:
                raise FileNotFoundError(f'No speech in supplied --clean-dir {clean}')
            clean = corpus
        if not _has_audio(speakers):
            if speakers_dir:
                raise FileNotFoundError(f'No speech in supplied --speakers-dir {speakers}')
            speakers = corpus
    return clean, speakers


def _finish(seconds):
    return (datetime.now().astimezone() + timedelta(seconds=seconds)).isoformat(timespec='seconds')


def train_models(models=None, *, preset='auto', clean_dir=None, speakers_dir=None,
                 noise_dir=None, estimate_only=False, install=True, download=True,
                 epochs=None, steps_per_epoch=None, val_items=None, max_hours=None,
                 device=None, resume=False, output_dir=None):
    """Prepare and train selected models sequentially; None means all local models.

    Returns a plan with measured ETAs and per-model completion status. CUDA uses
    the base presets. CPU uses a compact 16 kHz denoiser and tiny separator.
    Existing checkpoints are copied to a timestamped backup before new training.
    """
    models = select_models(models)
    if preset not in ('auto', 'tiny', 'compact', 'base'):
        raise ValueError('preset must be auto, tiny, compact, or base')
    for name, value in (('epochs', epochs), ('steps_per_epoch', steps_per_epoch), ('val_items', val_items)):
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive')
    if max_hours is not None and max_hours <= 0:
        raise ValueError('max_hours must be positive')
    python = prepare_environment(install)
    clean, speakers = prepare_data(models, clean_dir, speakers_dir, download)
    folder = ROOT / 'runs' / ('setup_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=True)
    out = Path(output_dir).resolve() if output_dir else ROOT / 'models/checkpoints'
    cuda = _run([python, '-c', 'import torch; print(torch.cuda.is_available())'], capture=True).stdout.strip() == 'True'
    hardware_device = device or ('cuda' if cuda else 'cpu')
    plan = dict(models=models, device=hardware_device, status='estimating', queue=[], started_at=_finish(0))
    plan_path = folder / 'plan.json'
    _write_plan(plan_path, plan)
    commands = []
    for model in models:
        chosen = preset
        if chosen == 'auto':
            chosen = 'base' if hardware_device.startswith('cuda') else ('compact' if model == 'local_unet' else 'tiny')
        if chosen == 'compact' and model == 'local_convtasnet':
            chosen = 'tiny'
        defaults = _run([python, '-c', f"import json; from {TRAINERS[model]} import DEFAULTS, PRESETS; print(json.dumps(dict(DEFAULTS, **PRESETS['{chosen}'])))"], capture=True)
        cfg = json.loads(defaults.stdout)
        cfg.update(clean_dir=str(clean if model == 'local_unet' else speakers), out_dir=str(out),
                   noise_dir=str(Path(noise_dir).resolve()) if noise_dir else None)
        for key, value in (('epochs', epochs), ('steps_per_epoch', steps_per_epoch), ('val_items', val_items)):
            if value is not None:
                cfg[key] = value
        stem = 'denoise_unet' if model == 'local_unet' else 'separation_convtasnet'
        command = [python, '-m', TRAINERS[model], '--config', folder / f'{model}.json',
                   '--device', hardware_device, '--estimate-json', folder / f'{model}_eta.json']
        checkpoint = out / f'{stem}_last.pt'
        if resume and checkpoint.exists():
            # Checkpoint architecture/rate must win over hardware shortcuts.
            metadata = _run([python, '-c', "import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps(c['config']))", checkpoint], capture=True)
            saved = json.loads(metadata.stdout)
            saved.update(clean_dir=cfg['clean_dir'], noise_dir=cfg['noise_dir'], out_dir=str(out))
            for key, value in (('epochs', epochs), ('steps_per_epoch', steps_per_epoch), ('val_items', val_items)):
                if value is not None:
                    saved[key] = value
            cfg = saved
            command += ['--resume', checkpoint]
        (folder / f'{model}.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        print(f'\nMeasuring {model} on {hardware_device} ({chosen})...', flush=True)
        probe_start = time.monotonic()
        _run(command + ['--estimate-only'])
        startup_seconds = time.monotonic() - probe_start
        estimate = json.loads((folder / f'{model}_eta.json').read_text(encoding='utf-8'))
        seconds = estimate['remaining_seconds']
        if max_hours:
            seconds = min(seconds, max_hours * 3600 + estimate['validation_seconds_per_epoch'])
        plan['queue'].append(dict(model=model, estimated_seconds=seconds + startup_seconds,
                                  startup_seconds=startup_seconds, measured_estimate=estimate,
                                  status='pending', checkpoint=str(out / f'{stem}_best.pt')))
        commands.append((model, command, stem))
    cumulative = 0.
    print('\nMeasured training queue (validation included):', flush=True)
    for entry in plan['queue']:
        cumulative += entry['estimated_seconds']
        entry['estimated_finish'] = _finish(cumulative)
        print(f"  {entry['model']}: {entry['estimated_seconds'] / 3600:.2f} hours; finish {entry['estimated_finish']}", flush=True)
    plan.update(status='estimated' if estimate_only else 'training', estimated_finish=_finish(cumulative))
    _write_plan(plan_path, plan)
    print(f'Queue and ETAs saved to {plan_path}', flush=True)
    if estimate_only:
        return plan
    try:
        for index, (model, command, stem) in enumerate(commands):
            entry = plan['queue'][index]
            entry.update(status='training', started_at=_finish(0))
            _write_plan(plan_path, plan)
            if not resume:
                for suffix in ('best', 'last'):
                    existing = out / f'{stem}_{suffix}.pt'
                    if existing.exists():
                        backup = folder / 'previous_checkpoints'
                        backup.mkdir(exist_ok=True)
                        shutil.copy2(existing, backup / existing.name)
            if max_hours:
                command += ['--max-hours', str(max_hours)]
            _run(command)
            final = json.loads((folder / f'{model}_eta.json').read_text(encoding='utf-8'))
            entry.update(status='stopped_early' if final.get('stopped_early') else 'complete',
                         finished_at=_finish(0), training_report=final)
            remaining = 0.
            for waiting in plan['queue'][index + 1:]:
                remaining += waiting['estimated_seconds']
                waiting['estimated_finish'] = _finish(remaining)
            plan['estimated_finish'] = _finish(remaining)
            _write_plan(plan_path, plan)
    except (Exception, KeyboardInterrupt) as exc:
        entry.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        plan['status'] = entry['status']
        _write_plan(plan_path, plan)
        raise
    plan['status'] = 'stopped_early' if any(e['status'] == 'stopped_early' for e in plan['queue']) else 'complete'
    _write_plan(plan_path, plan)
    print(f"\nTraining queue {plan['status']}. Checkpoints: {out}", flush=True)
    return plan


def _write_plan(path, plan):
    path.write_text(json.dumps(plan, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='all')
    parser.add_argument('--preset', choices=('auto', 'tiny', 'compact', 'base'), default='auto')
    parser.add_argument('--clean-dir')
    parser.add_argument('--speakers-dir')
    parser.add_argument('--noise-dir')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps-per-epoch', type=int)
    parser.add_argument('--val-items', type=int)
    parser.add_argument('--max-hours', type=float, help='time budget per model; saves a partial checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--estimate-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-install', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    args = vars(parser.parse_args())
    args['install'], args['download'] = not args.pop('no_install'), not args.pop('no_download')
    try:
        train_models(**args)
    except KeyboardInterrupt:
        print('\nTraining interrupted; rerun with --resume to use saved checkpoints.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'\nSetup failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
