# Stage 3: speech to text

Each speaker's separated audio gets its own transcript, preview and **Download
text file** button. Text files are plain UTF-8 (including Persian), and the
per-path / whole-job ZIP downloads include them. This is transcription in the
original language, not translation to English.

## Install and use

```bash
pip install -r requirements-stt.txt
python run.py serve
```

In Options choose a **Speech-to-text method** and **Speech language**. Auto
prefers installed Faster Whisper, then Transformers Whisper. Pin Persian
(`fa`) when you know the recording's language; automatic detection on very short
or noisy turns can be wrong. The row builder can override the recognizer for
each path, so you can compare algorithms on the same recording. Its default
inherits the job-wide selection. Choose **No transcription** for audio only.

| Method key | Default model | Languages | Trade-off |
|---|---|---|---|
| `faster_whisper` | multilingual Whisper small, CPU INT8 | 99 supported Whisper languages, including Persian | Default local choice; segment timestamps; ~500 MB download |
| `whisper_transformers` | `openai/whisper-small`, PyTorch | Same supported Whisper languages | Alternative implementation; more memory; ~1 GB download |
| `persian_wav2vec2` | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | Persian only | Persian-trained CTC alternative; ~1.2 GB; limited punctuation |
| `vosk` | small English 0.15 / Persian 0.42 | English or Persian, selected explicitly | Low-memory CPU model, ~40 / 53 MB; limited punctuation |

The language selector disables unsupported languages for a selected engine.
The Persian specialist's Auto setting means **assume Persian**, not language
detection. Vosk requires an explicit language. A Persian-specific model is not
automatically better than Whisper on every dialect or recording. No backend
supports literally every language; the UI lists the supported choices.

All four engines run locally. First use downloads model weights, never uploads
your recording. There is no ASR training step: these are pretrained models.
Root `setup.py` remains the trainer for the existing two local audio models.

```bash
# Optional: download ahead of time for offline recognition
python scripts/download_models.py faster_whisper vosk_en vosk_fa
python scripts/download_models.py whisper_transformers persian_wav2vec2

# Standalone audio or video transcription
python run.py transcribe meeting.wav --method faster_whisper --language fa -o meeting.txt
python run.py transcribe meeting.mp4 --method vosk --language en --json

# Full pipeline: each speaker gets WAV + TXT
python run.py pipeline meeting.wav --paths path1,path8 --speakers 2 --transcriber faster_whisper --language fa
```

Programmatic API:

```python
from src.transcription import transcribe
from src.pipeline import PipelineOptions, resolve_path, run_pipeline

result = transcribe('speaker.wav', method='faster_whisper', language='fa', output_path='speaker.txt')
report = run_pipeline('meeting.wav', [resolve_path({'id': 'path1'})],
    PipelineOptions(transcription_method='auto', transcription_language='fa'))
```

GUI and CLI pipeline default to Auto. For backwards compatibility,
`PipelineOptions()` and API requests omitting transcription options remain
audio-only; explicitly send `options.transcription_method` and
`options.transcription_language` to `/api/jobs`. Paths can include a
`transcriber` key (`""` inherits, `none` disables, or an engine key / `auto`).

## Models and hardware

Weights live under `models/pretrained/stt/` and are ignored by Git. Another
computer needs the dependencies and either internet on first run or a copied
model cache. Environment variables (set before starting the server):

| Variable | Meaning |
|---|---|
| `STT_WHISPER_MODEL` | Faster Whisper size (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`) or converted CTranslate2 directory |
| `STT_COMPUTE_TYPE` | Faster Whisper precision; default INT8 on CPU, INT8/float16 on CUDA |
| `STT_HF_WHISPER_MODEL` | Transformers Whisper repository or local directory |
| `STT_PERSIAN_MODEL` | Compatible Persian CTC repository or local directory |
| `STT_DEVICE` | `cpu`, `cuda`, or `auto`; Faster Whisper defaults to CPU for reliable Windows operation |
| `STT_VOSK_MODEL_DIR` | Extracted custom English/Persian Vosk model directory |
| `STT_VOSK_LANGUAGE` | Required language (`en` / `fa`) matching that custom Vosk model |
| `DS_THREADS` | CPU thread cap shared with audio backends |

Faster Whisper CUDA needs compatible CUDA/cuDNN libraries; detecting a GPU
does not prove these libraries are installed. CPU works without them. Larger
Whisper models can improve accuracy but need more memory and time. Do not use
an English-only `.en` checkpoint with Persian; the adapter rejects mismatches.

## Failure handling and time alignment

Audio is exported before recognition. If dependencies, model loading,
recognition or text export fails, the speaker WAV stays downloadable and the UI
shows the reason instead of an invented transcript. `tracks[].transcription`
contains `status`, `text`, `language`, `model_id`, `segments`, `elapsed`, error
details and, on success, the text filename. Path audio status is independent of
`path.transcription.status` (`ok`, `partial`, `failed`, `disabled`).

Recognition uses each speaker's existing turns, with short context padding,
bounded windows and quiet-boundary cuts for long turns. Disjoint turns are not
joined into a misleading timeline. Whisper / Vosk provide model-derived
segment times; Transformers short-form Whisper and Persian CTC have coarse
window times, not forced word alignment. Silence does not trigger model loads
or fabricated words. Automatic language detection can differ across speakers.

Accuracy depends on the recording, denoising artifacts, overlap leakage,
language, dialect and model size. Diarization cannot remove simultaneous
speakers, so its transcripts can contain cross-talk. Transcripts should be
reviewed before important use; there is no promise of perfect recognition.
Highly repetitive output is flagged in the UI and transcript metrics for
review, not automatically discarded: real speech can also contain repetition.

## Verification

```bash
python -m pytest tests/test_transcription.py -q
node tests/test_frontend_transcription.cjs
python scripts/audit_transcription.py --methods faster_whisper,vosk --language en --label english
# Requires playwright and an installed Chrome; start the server first:
python scripts/check_transcription_ui.py --url http://127.0.0.1:8000
```

The unit tests use fake models: no internet or pretrained download is needed.
The audit script uses real recognition and reports WER/CER when a reference
transcript exists. The browser check covers selectors, supported-language
restrictions, per-path choices, HTML escaping, RTL text, actual UTF-8 file
download, and mobile layout. Audit outputs are under `data/outputs/stt_audit_*`.

Primary model/API references:
[Faster Whisper](https://github.com/SYSTRAN/faster-whisper),
[Transformers Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper),
[Persian Wav2Vec2 model card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian),
[Vosk models](https://alphacephei.com/vosk/models).
