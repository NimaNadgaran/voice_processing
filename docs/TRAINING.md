# Local model training and finish-time estimates

Root `setup.py` is a training launcher, **not** a setuptools installer. Running
it explicitly prepares the environment/data, estimates time on that computer,
and trains selected models sequentially. Starting the web UI never starts it.

## Fresh clone

```bash
python setup.py --estimate-only
python setup.py
```

Default order is `local_unet` (SpectralUNet denoiser), then `local_convtasnet`
(speaker separator). The script creates/reuses `.venv`, installs core/training
dependencies when its readiness probe finds them missing, and runs trainers
using that environment. Missing default training data triggers LibriSpeech
`train-clean-100` (~6.4 GB archive; about 15 GB free disk required).

`--estimate-only` prevents full training, but can still install packages,
download/extract data and run short timing/validation probes. It is not a
zero-cost command. For prepared offline data, use
`--no-install --no-download` and supply appropriate paths.

## Choose models and control the run

```bash
python setup.py --models local_unet
python setup.py --models local_convtasnet
python setup.py --models local_unet,local_convtasnet --resume
python setup.py --device cpu --max-hours 2
python setup.py --clean-dir data/datasets/clean --speakers-dir data/datasets/speakers --noise-dir data/datasets/noise
python setup.py --output-dir models/checkpoints --epochs 10 --steps-per-epoch 100 --val-items 32
```

| Argument | Meaning |
|---|---|
| `--models` | `all` (default), one key, or comma-separated keys |
| `--preset` | `auto`, `tiny`, `compact`, `base`; compact separator maps to tiny |
| `--device` | Explicit compute device; otherwise CUDA if usable, else CPU |
| `--estimate-only` | Prepare and estimate without the full training queue |
| `--epochs`, `--steps-per-epoch`, `--val-items` | Positive overrides; change time and quality expectations |
| `--max-hours` | Positive time budget **per model**, not for the whole queue |
| `--resume` | Restore the saved checkpoint/configuration and training state |
| `--output-dir` | Checkpoint destination; default `models/checkpoints/` |
| `--clean-dir`, `--speakers-dir`, `--noise-dir` | Existing training speech / per-speaker speech / optional noise |
| `--no-install`, `--no-download` | Fail on missing preparation instead of fetching it |

Programmatic entry point:

```python
from setup import train_models

plan = train_models()  # models=None: all local trainable models, one after another
plan = train_models(models=['local_unet'], estimate_only=True)
```

Pretrained DeepFilterNet, downloaded separators and speech-to-text models do
not belong to this training queue. For their weights use
`scripts/download_models.py`, not `setup.py`.

## ETA: what the finish time includes

Estimates measure actual model steps on the selected hardware, including data
loading and validation. Timing probes restore model/optimizer/RNG state and do
not publish probe weights as trained checkpoints. The console prints per-model
and cumulative finish estimates; per-epoch updates track actual progress.

Dependencies and corpus preparation happen before training ETAs. Corpus
downloads report progress, speed and estimated remaining download time.
Hardware load, storage, batch sizes and validation cost can change the ETA;
there is no guaranteed completion deadline. Do not use another machine's
hour estimate as your own measurement.

CPU Auto uses a compact 16 kHz denoiser and tiny separator; CUDA Auto uses base
presets. Exact effective settings, ETA reports and queue status are saved in
`runs/setup_<timestamp>_<id>/`, including `plan.json`. These outputs are ignored
by Git and are not present in a fresh clone.

## Data and evaluation

Clean speech can be a WAV/FLAC tree. Separator speech should preserve speaker
identity in separate directories; organized LibriSpeech is supported. Missing
optional noise uses synthesized corruption for the denoiser. Explicit invalid
data paths fail instead of silently fetching a different corpus. Automatic
reuse excludes evaluation corpora; do not train on held-out test speech.

Existing training data can be reused for both models without duplicating it;
see [data/README.md](../data/README.md) and the stage training sections:
[denoising](../src/denoising/README.md),
[separation](../src/separation/README.md).

## Checkpoints and interrupted runs

The application defaults to:

- `models/checkpoints/denoise_unet_best.pt`
- `models/checkpoints/separation_convtasnet_best.pt`

Fresh training backs up existing best/last checkpoints in the run folder.
`--resume` restores configuration, optimizer/scheduler/scaler and best score.
Time-budget stops save partial state; resume repeats an unfinished epoch.
Ctrl+C can only resume from the latest saved checkpoint, not necessarily from
the interrupted step. Keep both checkpoints and run plans when moving work to
another computer; `.gitignore` deliberately does not publish them.

Only load checkpoints from sources you trust. A completed optimization run
does not prove good audio quality. The existing small setup/resume verification
was a functional test, not full quality training:
[AUDIO_VALIDATION.md](../AUDIO_VALIDATION.md).
