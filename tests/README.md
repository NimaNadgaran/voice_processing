# Tests and verification

The last complete local suite passed **168 tests** with one Starlette
TestClient deprecation warning. That result is a recorded verification, not a
promise for every optional dependency/version/platform. Details are in
[audio validation](../AUDIO_VALIDATION.md) and
[transcription validation](../TRANSCRIPTION_VALIDATION.md).

## Fast checks

From the project root, with core requirements installed:

```bash
python tests/test_smoke.py
node --check frontend/app.js
node tests/test_frontend_transcription.cjs
```

Node is needed for the JS checks, **not** to run/build the application UI.

For the expanded suite install test tools into the same environment:

```bash
python -m pip install pytest httpx
python -m pytest tests -q
```

Backend/CTC adapter tests also use numerical/model packages such as Torch.
Use the application/STT/training requirements appropriate to the tests rather
than assuming every optional backend is installed. Unit-model adapters are
mocked: those tests do not download ASR weights or upload recordings.

| File | Coverage |
|---|---|
| `test_smoke.py` | Core audio/DSP/registry, speaker count and basic pipelines |
| `test_regressions.py` | Input/options/path validation and earlier core/API regressions |
| `test_audio_backends.py` | Backend shapes, rates/chunks, stitching, failures and training state / ETA behavior |
| `test_transcription.py` | Languages/models, silence/chunks/timestamps, safe exports, independent audio success, UTF-8 TXT/ZIP API |
| `test_frontend_transcription.cjs` | Pure rendering, escaping, RTL, warnings and text-download states |

## Real inference is a separate check

```bash
python scripts/audit_audio.py --save-audio --label verified
python scripts/audit_transcription.py --methods faster_whisper,vosk --language en --label english
python run.py serve --port 9139
# Another terminal:
python scripts/selftest.py --url http://127.0.0.1:9139
python scripts/check_transcription_ui.py --url http://127.0.0.1:9139
```

Real audits require packages, weights and/or recorded reference corpora. The
Chrome script additionally needs Playwright and installed Chrome; it saves
screenshots/download evidence under `data/outputs/`. Missing gated-model access
is not equivalent to failed audio quality, and valid finite audio is not proof
of clean separation. Inspect real text against reference speech, especially
Persian/dialectal/noisy audio. Do not use UI reference-free scores as ground truth.

Test caches, reports, recordings and weights are ignored. Keep reproducible
commands and summaries in Markdown; do not commit private/generated data merely
to publish test evidence. For temporary-path restrictions use a writable,
dedicated test directory rather than deleting shared folders or user data.
