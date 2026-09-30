# Speech-to-text implementation and verification

Checked locally on 2026-09-30, Windows, Python 3.14, CPU inference.

## Implemented

Third-stage recognition runs on each separated speaker, not on the whole mixed
recording. Four selectable local algorithms, language metadata / restrictions,
per-path overrides, Unicode previews, per-speaker **Download text file** links,
text in ZIPs, timestamped JSON, and standalone CLI are integrated. Disabled,
empty, failed and partial recognition are distinct from audio processing status.
Already produced audio remains usable when recognition or text export fails.

Dependencies / models are optional; the UI tells users what to install. Auto
does not send audio to a hosted API or assume Persian for unknown languages.
See [Stage 3 usage](src/transcription/README.md).

## Automated checks

Final full-suite result: **168 passed** (one unrelated Starlette TestClient
deprecation warning). JS syntax/rendering and the real Chrome checks also passed.

The full pytest suite includes language aliases and restrictions, model
generator exhaustion, transcribe versus translate, resampling / chunk bounds,
original timeline offsets, silence handling, clearing pinned language for Auto,
English-only model restrictions, CUDA-library failures, safe file/export
failures, independent audio success, UTF-8 downloads and text inside ZIPs.

Node checks verify JS syntax, safe rendering/escaping, RTL direction, warnings
and download/failure states. Real headless Chrome checks verify all four method
cards, global and per-path language compatibility, Persian UTF-8 downloads,
script escaping and 390-pixel mobile layout.

## Actual model recognition

Models tested, not just mocked:

| Engine | Actual tested weights | English | Persian |
|---|---|---|---|
| Faster Whisper | multilingual `small`, CPU INT8 | Recognized the reference utterance; Auto detected English | Produced Persian text; Auto detected Persian |
| Transformers Whisper | `openai/whisper-tiny`, CPU | Recognized the reference utterance; Auto reported English correctly | Inference/export worked, but this tiny model gave poor/repetitive output on the vlog clip |
| Persian Wav2Vec2 | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | Intentionally rejected: Persian-only model | Loaded the actual ~1.2 GB checkpoint and produced Persian text |
| Vosk | English small 0.15 / Persian small 0.42 | Recognized the reference utterance | Downloaded/loaded the correct Persian model and produced Persian text |

The English LibriSpeech reference utterance is 14.65 seconds. Measured WER:
Faster Whisper 0.200, Vosk 0.133, Transformers tiny 0.222. These are **single
utterance results, not general model rankings**. Number formatting (Whisper's
`1845` versus reference `eighteen forty five`) contributes to WER here; the
normalizer does not equate those forms. This is not proof that Vosk is generally
more accurate than Whisper.

The Persian test was a 12-second clip from a bundled vlog screen recording.
It has no human-verified reference, so no Persian WER / accuracy claim is made.
Results varied and should be reviewed against the recording. Tiny Whisper is
not recommended as a Persian-quality preset; the Transformers backend defaults
to `openai/whisper-small`, and larger multilingual models can be configured.
That default Transformers small checkpoint was **not** separately benchmarked
here. A Persian-trained specialist is not guaranteed to beat Whisper on every
recording. Repetition suppression and quality warnings cannot fix all model
hallucinations or dialect mismatch.

Real-testing fixes included CPU INT8 as the reliable Faster Whisper default on
this Windows machine (CUDA libraries were missing despite a visible GPU), an
actionable CPU fix for lazy CUDA generator failures, clearing pinned language
configuration, reading detected language from the returned decoder prefix,
bounded generation with a repetition constraint, correct Persian Vosk 0.42
weights, and mobile audio/select widths. Automatic remote checkpoint conversion
is disabled by default for the legacy Persian checkpoint; it is unnecessary
for local recognition. Torch >=2.6 is required for safer legacy checkpoint
loading.

## End-to-end evidence

The CLI pipeline processed a real 9.1-second two-speaker recording through
`path1` and `path8`, producing two WAVs and two nonempty TXT files per path.
Report: `data/outputs/job_20260930-211721_21a510/report.json`.

A real Chrome upload ran `path1`, waited for completion, downloaded both
speaker transcripts, and downloaded/read the path ZIP with both text files.
No model mocks were used for that run. Evidence (desktop, mobile, real pipeline
screenshots, downloaded UTF-8 text and ZIP):
`data/outputs/stt_ui_verification/`.

Real audit reports and exported text:

- `data/outputs/stt_audit_english_verified/`
- `data/outputs/stt_audit_persian_verified/`
- `data/outputs/stt_audit_persian_specialist_verified/`
- `data/outputs/stt_audit_transformers_english_verified/`
- `data/outputs/stt_audit_auto_language_verified/`
- `data/outputs/stt_audit_transformers_auto_verified/`

Generated recordings, downloaded weights, screenshots and test outputs are not
source files and are ignored by Git. No ASR training was needed or started.

## Reproduce

```bash
pip install -r requirements-stt.txt
python -m pytest tests -q
node --check frontend/app.js
node tests/test_frontend_transcription.cjs
python scripts/audit_transcription.py --methods faster_whisper,vosk --language en --label english_verified
python scripts/audit_transcription.py --input your_persian.wav --language fa --methods faster_whisper,persian_wav2vec2,vosk --reference-text "your verified transcript" --label persian_reference
python run.py serve --port 9139
# Separate terminal; requires playwright and Chrome:
python scripts/check_transcription_ui.py --url http://127.0.0.1:9139 --run-pipeline data/outputs/path_audit_final_overlap_preset/turn_taking.wav
```
