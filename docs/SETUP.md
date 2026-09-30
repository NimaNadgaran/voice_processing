# Fresh-clone setup and everyday use

Start at the repository root. Source files, configs and the synthetic UI demo
are published; environments, personal audio, downloaded models, datasets and
previous output files are deliberately not included.

## 1. Create an environment

Use a Python version supported by the optional packages you choose. The
validated local environment used Python 3.14, but some audio packages require
older interpreters. `python run.py doctor` reports actual readiness. The
standalone DeepFilterNet route avoids its Python-extension compatibility limit.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-stt.txt
.\.venv\Scripts\python.exe scripts/download_models.py deepfilternet faster_whisper
.\.venv\Scripts\python.exe run.py serve
```

Activation is optional when using the environment's interpreter directly. The
included `run.ps1` uses `.venv\Scripts\python.exe`:

```powershell
.\run.ps1 doctor
.\run.ps1 serve
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-stt.txt
python scripts/download_models.py deepfilternet faster_whisper
python run.py serve
```

Open the URL printed by the server (normally `http://127.0.0.1:8000`). A default
busy port rolls to the next free port; explicitly passing `--port 8080` requests
that exact port and reports a conflict instead of silently moving.

## 2. Suggested workflow

The suggested audio algorithms are **DeepFilterNet (`deepfilternet`)** and
**built-in Clustering Diarization (`diarize_cluster`)**.

1. Upload an audio or video recording.
2. Build a custom row with DeepFilterNet and built-in clustering. Tick that row;
   untick other paths if you only want one output set.
3. Select Faster Whisper / another speech-to-text engine and the spoken
   language. Persian is `fa`; multilingual Whisper also supports Auto.
4. Pin the speaker count when known, then run.
5. Download the denoised audio, each speaker's WAV, each **Download text file**,
   or the per-path / whole-job ZIP.

`path1` is the lightweight spectral-gate/clustering preset. `path2` names
pyannote, not the suggested built-in separator. Preset fallbacks are recorded
in the report. Clustering attributes turns; it cannot unmix simultaneous voices.

CLI example (after activating the environment, or via its interpreter):

```bash
python run.py pipeline meeting.wav --denoiser deepfilternet --separator diarize_cluster --transcriber faster_whisper --language fa
python run.py pipeline meeting.wav --paths path8 --transcriber none
python run.py transcribe speaker.wav --method vosk --language en -o speaker.txt --json
```

## 3. Choose dependencies deliberately

| File | Purpose |
|---|---|
| `requirements.txt` | Core DSP, built-in clustering, web UI, bundled ffmpeg support |
| `requirements-stt.txt` | All four local speech-to-text adapters; includes Torch >=2.6 |
| `requirements-optional.txt` | Optional audio packages; read compatibility notes before installing everything |
| `requirements-train.txt` | Local denoiser / separator training |

For a smaller speech-to-text installation, install `faster-whisper` alongside
core requirements and use Faster Whisper; missing alternative engines remain
visible but disabled. For audio only, install core requirements, download
DeepFilterNet, and choose **No speech-to-text** in the UI. Missing ASR models do
not prevent already produced speaker audio from being downloaded.

The standalone DeepFilterNet executable downloads only for supported
platform/architecture combinations. Set `DEEP_FILTER_BIN` if you have a local
compatible binary. Some optional packages use their own Torch/Hugging Face
caches; the project ignores its local copies, not your entire user cache.

## 4. Models, offline use and training

```bash
python run.py doctor
python run.py methods
python scripts/download_models.py --list
python scripts/download_models.py faster_whisper vosk_en vosk_fa
```

`--list` reports download targets and installed packages, not proof that weights
are already cached. `--all` may download several GB and may encounter gated
models. Prefer explicit targets. First-use model download time is separate from
inference time; all STT engines recognize locally after downloading weights.

Copy required model caches separately for offline use; see
[models/README.md](../models/README.md). Language/size/device overrides are in
[the transcription guide](../src/transcription/README.md). CPU INT8 is the
reliable Faster Whisper default; a visible GPU alone does not establish CUDA
library compatibility.

Training is separate from app startup:

```bash
python setup.py --estimate-only
python setup.py
```

These commands can install dependencies and download training data. They train
only `local_unet` / `local_convtasnet`, not pretrained DeepFilterNet or STT models.
See [training and ETA](TRAINING.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing model / disabled method | `run.py doctor`, exact install/download hint, then restart the server |
| CUDA library failure in Whisper | Set `STT_DEVICE=cpu` before starting the server |
| Wrong Persian text | Pin `fa`, compare larger Whisper / Persian CTC, review the original audio |
| Two voices in a speaker track | Diarization does not remove overlapping voices; compare neural separation |
| Too many/few output speakers | Pin the known count; inspect the original and turn timeline |
| Video cannot decode | Check ffmpeg in `doctor`; core requirements include `imageio-ffmpeg` |
| Missing local checkpoints after cloning | Run training or copy your own trusted checkpoints separately |
| Training ETA unavailable yet | Dependencies/data must be ready before real timing probes run |

Keep the original recording. Metrics are estimates, recognition may hallucinate,
and the recommended algorithms are not universally best on every recording.
Detailed exceptions are in `data/cache/errors.log`; redact private details before
sharing logs. The default server is for localhost use, not public deployment.
