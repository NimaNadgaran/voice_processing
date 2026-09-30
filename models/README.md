# `models/` — weights live here

Nothing in this folder is committed to git (see `.gitignore`); it is filled in
by downloads or by your own training runs.

```
models/
├── checkpoints/     ← models YOU train
│   ├── denoise_unet_best.pt              (src/denoising/training/train_unet.py)
│   ├── denoise_unet_last.pt
│   ├── separation_convtasnet_best.pt     (src/separation/training/train_convtasnet.py)
│   └── separation_convtasnet_last.pt
└── pretrained/      ← third-party weights cached here on first use
    ├── ecapa/                            SpeechBrain speaker embeddings
    ├── speechbrain__sepformer-whamr16k/
    └── …
```

The architecture *code* lives next to its stage — `src/denoising/architectures.py`
and `src/separation/architectures.py` — so each stage stays self-contained.
This folder only holds weights.

---

## Checkpoints you train

| file | produced by | used by | contains |
|---|---|---|---|
| `denoise_unet_best.pt` | `python -m src.denoising.training.train_unet` | denoiser `local_unet` | `model` state dict, `config`, `epoch`, `val_si_sdr`, `params_m` |
| `separation_convtasnet_best.pt` | `python -m src.separation.training.train_convtasnet` | separator `local_convtasnet` | same shape, plus `val_si_sdri` |

The `config` block is what lets the inference wrapper rebuild the exact
architecture — you never have to keep the training config around.

Point the app at a different file with environment variables:

```bash
DENOISE_UNET_CKPT=/path/to/other.pt
SEPARATION_CKPT=/path/to/other.pt
```

---

## Pretrained weights (downloaded on demand)

Every backend downloads its own weights the first time it runs, into this folder
or the Hugging Face cache. Sizes are approximate.

| backend | model | size | licence | needs a token |
|---|---|---|---|---|
| `deepfilternet` | DeepFilterNet3 | ~2 MB | MIT / Apache-2.0 | no |
| `demucs_denoiser` | dns64 | ~130 MB | MIT | no |
| `demucs_vocals` | htdemucs | ~80 MB | MIT | no |
| `rnnoise` | bundled in the wheel | 85 kB | BSD | no |
| `resemble_enhance` | denoiser + enhancer | ~500 MB | MIT | no |
| `sepformer` | sepformer-whamr16k / wsj0-2mix / -3mix | ~110 MB each | Apache-2.0 | no |
| `convtasnet_asteroid` | ConvTasNet_Libri2Mix / Libri3Mix | ~20 MB each | MIT | no |
| `mossformer_clearvoice` | MossFormer2_SS_16K | ~200 MB | Apache-2.0 | no |
| `pyannote` | speaker-diarization-3.1 + segmentation-3.0 | ~30 MB | MIT, **gated** | **yes** (free) |
| `nemo_msdd` | diar_msdd_telephonic | ~90 MB | Apache-2.0 | no |
| speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` | ~80 MB | Apache-2.0 | no |

### pyannote is gated (but free)

1. Sign in at huggingface.co.
2. Accept the conditions on **both**
   `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
3. Create a read token and export it:

```bash
set HF_TOKEN=hf_xxx          # Windows
export HF_TOKEN=hf_xxx       # macOS / Linux
```

---

## Pre-downloading (for an offline machine)

```bash
python scripts/download_models.py --list          # what is installed / cached
python scripts/download_models.py --all           # everything that is installed
python scripts/download_models.py deepfilternet sepformer
```

The script only downloads backends whose Python package is already installed,
and it never starts training.

To move a cache to an offline machine, copy `models/pretrained/` **and** your
Hugging Face cache (`%USERPROFILE%\.cache\huggingface` / `~/.cache/huggingface`).

---

## Disk budget

| you install | disk |
|---|---|
| nothing extra (DSP methods only) | 0 MB |
| `deepfilternet` + `speechbrain` (recommended minimum) | ~200 MB |
| add `pyannote` | ~230 MB |
| everything except NeMo | ~1.2 GB |
| everything | ~4 GB |
| plus torch itself | ~2.5 GB (CPU wheel) / ~5 GB (CUDA) |
