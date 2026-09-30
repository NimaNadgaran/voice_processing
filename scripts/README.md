# Development and model utilities

Run scripts from the repository root using the project environment. Generated
audio, metrics, downloaded weights and benchmark reports are local artifacts,
not GitHub source. Use `--help` for exact arguments and dependency requirements.

| Script | Purpose / important boundary |
|---|---|
| `make_demo_audio.py` | Generates synthetic multi-speaker audio; no real private conversation required |
| `download_models.py` | Pre-downloads selected pretrained models/binaries; never trains or installs packages |
| `prepare_datasets.py` | Checks, links and organizes training data; root `setup.py` is the automatic training-data downloader |
| `benchmark.py` | Synthetic or recorded-reference evaluation of counting, denoising and separation |
| `benchmark_denoising.py` | Held-out recorded voices, noise types, rates and SNR conditions |
| `audit_audio.py` | Actual local backend inference, reference scores and availability; excludes cloud uploads |
| `benchmark_paths.py` | Full audio-path comparisons; gated models are opt-in and cloud calls excluded |
| `audit_transcription.py` | Actual STT inference, UTF-8 text export, optional reference WER/CER |
| `check_transcription_ui.py` | Headless Chrome selectors/download/escaping/mobile checks; optional real upload workflow |
| `selftest.py` | Exercises a running local server with audio/video uploads and file/ZIP checks |

## Common commands

```bash
python scripts/download_models.py --list
python scripts/download_models.py deepfilternet faster_whisper
python scripts/download_models.py vosk_en vosk_fa
python scripts/prepare_datasets.py --check
python scripts/benchmark.py --source synthetic
python scripts/audit_audio.py --save-audio --label verified
python scripts/benchmark_paths.py --paths path1,path3,path8 --label verified
python scripts/audit_transcription.py --methods faster_whisper,vosk --language en --label english
```

Recorded-reference audio audits default to local LibriSpeech `dev-clean` under
`data/datasets/_download/LibriSpeech/`; that corpus is not committed. Acquire
evaluation speech separately from training data and honor its license. Root
`setup.py` prepares `train-clean-100`, **not** this held-out evaluation corpus.
Models may download on first real use; package availability is not proof of
cached weights or access to gated models.

For a real transcript accuracy check, provide reference text matching the
selected clip. An explicit recording without a reference only verifies
inference/export, not accuracy. Number formatting and normalization affect
WER/CER; small single-utterance results are not general model rankings.

## Server and browser checks

```bash
python run.py serve --port 9139
# In another terminal:
python scripts/selftest.py --url http://127.0.0.1:9139
python -m pip install playwright
python scripts/check_transcription_ui.py --url http://127.0.0.1:9139
```

The browser script uses an installed Chrome; it does not download a browser.
Install the STT dependencies for its four-engine selector tests. To verify the
real two-speaker workflow, add `--run-pipeline <two-speaker.wav>` after
pre-downloading Faster Whisper; use a recording with two distinct turn-taking
speakers. Synthetic tone demos do not establish real speech recognition quality.

See [tests/README.md](../tests/README.md),
[AUDIO_VALIDATION.md](../AUDIO_VALIDATION.md) and
[TRANSCRIPTION_VALIDATION.md](../TRANSCRIPTION_VALIDATION.md) for measured checks
and limitations. Training/ETA commands are documented in
[docs/TRAINING.md](../docs/TRAINING.md), not hidden inside these audits.
