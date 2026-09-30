# `models/` — weights live here

Only this guide and the empty-directory `.gitkeep` markers are committed.
Weights/binaries are filled in by downloads or your training runs and are
excluded by `.gitignore`; they are intentionally absent from a fresh clone.

The suggested audio stack is DeepFilterNet plus built-in clustering. Download
DeepFilterNet's supported standalone binary and Faster Whisper for that stack
with speech-to-text:

```bash
python scripts/download_models.py deepfilternet faster_whisper
```

Built-in MFCC clustering needs no pretrained weights. Optional SpeechBrain
ECAPA embeddings require an additional download.

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
    ├── stt/
    │   ├── faster_whisper/               converted local Whisper weights
    │   ├── huggingface/                  Transformers Whisper / Persian CTC
    │   └── vosk/                         separate English / Persian models
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

Those lines name variables; set them using your shell's syntax before starting
the app. For example in PowerShell: `$env:DENOISE_UNET_CKPT='D:\models\my.pt'`.
Use `python setup.py` to train both local models sequentially and measure their
finish-time estimates; see [docs/TRAINING.md](../docs/TRAINING.md). Best/last
checkpoints, backups and plans are not published. Only load checkpoints you trust.

---

## Pretrained weights (downloaded on demand)

Every backend downloads its own weights the first time it runs, into this folder
or the Hugging Face cache. Sizes are approximate.

| backend | model | size | licence | needs a token |
|---|---|---|---|---|
| `deepfilternet` | DeepFilterNet3 / supported standalone binary | binary ~27 MB | see upstream release | no |
| `demucs_denoiser` | dns64 | ~130 MB | MIT | no |
| `demucs_vocals` | htdemucs | ~80 MB | MIT | no |
| `rnnoise` | bundled in the wheel | 85 kB | BSD | no |
| `resemble_enhance` | denoiser + enhancer | ~500 MB | MIT | no |
| `sepformer` | sepformer-whamr16k / wsj0-2mix / -3mix | ~110 MB each | Apache-2.0 | no |
| `convtasnet_asteroid` | ConvTasNet_Libri2Mix / Libri3Mix | ~20 MB each | MIT | no |
| `mossformer_clearvoice` | MossFormer2_SS_16K | ~200 MB | Apache-2.0 | no |
| `pyannote` | version-matched community-1 (4.x) / 3.1 (3.x) | varies | see selected model conditions, **gated** | **yes** |
| `nemo_msdd` | diar_msdd_telephonic | ~90 MB | Apache-2.0 | no |
| speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` | ~80 MB | Apache-2.0 | no |
| `faster_whisper` | multilingual small, CPU INT8 | ~500 MB | see Whisper/upstream models | no |
| `whisper_transformers` | `openai/whisper-small` | ~1 GB | see model card | no |
| `persian_wav2vec2` | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | ~1.2 GB | see model card | no |
| `vosk` | small English 0.15 / Persian 0.42 | ~40 / 53 MB | see selected model listing | no |

Sizes are default-model estimates, not an exact disk plan. Packages, models and
training corpora have separate licenses; review the selected upstream release /
model card before redistribution or commercial use. The table is not a grant
of rights. Speech-to-text language, device and model overrides are documented in
[src/transcription/README.md](../src/transcription/README.md).

### pyannote requires model access

1. Sign in at huggingface.co.
2. For pyannote.audio **4.x**, accept the conditions for
   `pyannote/speaker-diarization-community-1`. For **3.x**, accept
   `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
3. Create a read token and export it:

```bash
set HF_TOKEN=hf_xxx          # Windows cmd (placeholder only)
export HF_TOKEN=hf_xxx       # macOS / Linux
```

PowerShell uses `$env:HF_TOKEN='your-local-token'`. Never commit that value.
Installed packages and a token do not prove access; accept conditions using
the same account. The suggested built-in clustering separator needs none of
this gating setup.

---

## Pre-downloading (for an offline machine)

```bash
python scripts/download_models.py --list          # targets / package availability, not cache verification
python scripts/download_models.py --all           # everything that is installed
python scripts/download_models.py deepfilternet sepformer
```

The script only downloads backends whose Python package is already installed,
and it never starts training. The standalone DeepFilterNet download needs no
DeepFilterNet Python package. `--all` includes large STT models and potentially
gated audio models; select explicit targets when you want a small/offline setup.

To move a cache to an offline machine, copy `models/pretrained/` **and** your
Hugging Face cache (`%USERPROFILE%\.cache\huggingface` / `~/.cache/huggingface`).
Some backends also use Torch's own cache. Copy only the caches required by the
selected algorithms, install compatible packages, and run a small local check
on the destination before relying on offline operation. Do not upload these
folders to GitHub merely to reproduce the source checkout.

---

## Disk budget

| you install | disk |
|---|---|
| nothing extra (DSP methods only) | 0 MB |
| standalone DeepFilterNet + built-in clustering | ~27 MB for the denoiser binary; no mandatory embedding model |
| add Faster Whisper small | ~500 MB model cache |
| add Transformers small / Persian CTC / both Vosk models | ~2.3 GB additional model caches |
| optional ECAPA | ~80 MB additional |
| local training corpus | ~6.4 GB archive plus extraction; allow ~15 GB free |

Python packages, Torch, temporary downloads, duplicate caches and other optional
audio models require additional disk space. GPU packages can be substantially
larger than CPU packages; larger Whisper variants increase the model budget.
