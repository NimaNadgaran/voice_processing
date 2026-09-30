# API and pipeline integration

Start the local server with `python run.py serve`. The web UI is served at `/`;
FastAPI's schema is available at `/docs` and `/openapi.json`. There is no separate
frontend build, API key or login layer.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Python/package readiness, stage method counts, ffmpeg and upload limit |
| `GET /api/methods` | Denoisers, separators, transcribers, language support and dependency hints |
| `GET /api/paths` | Presets with availability, effective methods and substitutions |
| `POST /api/jobs` | Multipart recording upload and background processing |
| `GET /api/jobs` | Recent in-process job metadata |
| `GET /api/jobs/{job_id}?since=0` | Job status, progress/events and report when completed |
| `GET /api/jobs/{job_id}/events?since=0` | Incremental event polling |
| `GET /api/jobs/{job_id}/report` | Full JSON report (including persisted completed reports) |
| `GET /api/jobs/{job_id}/zip` | Whole-job outputs |
| `GET /api/jobs/{job_id}/zip?path_id=custom_1` | One path's outputs |
| `GET /api/files/{job_id}/{path_id}/{filename}` | Speaker audio or UTF-8 TXT download |

## Submit a three-stage job

`file` is the audio/video upload; `paths` and `options` are JSON-encoded form
strings, not separate JSON request bodies. Python example:

```python
import json
import requests

paths = [{'id': 'custom_1', 'name': 'Suggested audio stack',
          'denoiser': 'deepfilternet', 'separator': 'diarize_cluster'}]
options = {'num_speakers': 2, 'transcription_method': 'faster_whisper',
           'transcription_language': 'fa'}
with open('meeting.wav', 'rb') as recording:
    response = requests.post('http://127.0.0.1:8000/api/jobs',
        files={'file': ('meeting.wav', recording, 'audio/wav')},
        data={'paths': json.dumps(paths), 'options': json.dumps(options)}, timeout=120)
response.raise_for_status()
job = response.json()
print(job['job_id'])
```

Use `num_speakers` only when known; omit it for detection. Read `next_seq` in
poll responses and send it as `since` so old events are not requested repeatedly.
Events include stage, percentage, message and sometimes a completed artifact.
The job's `queued/running/done/error` state is separate from per-path success.
A completed job may contain failed audio paths or failed/partial transcription;
inspect the report rather than interpreting `done` as an accuracy guarantee.

## Options and overrides

`PipelineOptions` validates speaker counts, count source, track normalization,
waveform settings, transcription method and language. Unknown options/methods
and unsafe or duplicate path IDs are rejected. Use method keys from
`/api/methods`, and respect each transcriber's supported languages.

Paths can specify preset IDs or custom denoiser/separator combinations. An
optional `transcriber` field overrides the job setting: empty string inherits,
`none` disables, `auto` chooses an installed compatible method, or select a
specific engine key. Language is job-wide; it must be compatible with all
selected engines. Vosk needs explicit `en` or `fa`; Persian CTC assumes Persian
when used with Auto. Whisper Auto can detect each output voice's language.

GUI / CLI pipeline default to Auto transcription. For backward compatibility,
API jobs that omit `transcription_method`, and `PipelineOptions()` in Python,
remain audio-only. Explicitly enable transcription as in the example above.

## Reports and files

The report includes input information, speaker estimation, selected options,
paths, timings and reference-free comparison metrics. Each path has denoised
audio, speaker tracks and `transcription` status. Each track's transcription
contains text, language, model ID, segment times, duration, warnings/errors and
the text filename when export succeeds.

Audio is saved before recognition. Missing ASR dependencies, failed model
loads, inference or TXT exports do not remove successfully exported speaker
WAVs. STT status can be `ok`, `empty`, `failed` or `disabled`; the path's STT
summary can also be `partial`. Empty recognized speech creates an empty TXT,
not fabricated text. Both path and whole-job ZIPs include produced TXT files.

File paths are validated within the output directory. TXT uses
`text/plain; charset=utf-8`; audio players can request WAV byte ranges. Timeline
offsets refer to the original recording. Transformers Whisper / Persian CTC
timestamps are coarse window times, not forced word alignment.

## Runtime and deployment boundaries

The default server binds to `127.0.0.1`. It has no authentication and permits
cross-origin API requests; do **not** expose it directly to untrusted users or
private recordings on a public network. Authentication, HTTPS, authorization,
storage isolation and retention would need a separate deployment design.

Uploads are streamed with `MAX_UPLOAD_MB` (default 1024). Jobs use two
in-process worker threads, with paths sequential within each job. Live job
state does not survive server restart; completed files/reports remain on disk.
Old recordings and output folders are not automatically deleted.

All speech-to-text engines run locally after weight downloads. Only explicitly
configured `api_huggingface` denoise/separation methods send recordings off the
machine; they require tokens and compatible deployed endpoints. Never commit
real tokens, upload private audio as test data, or publish unredacted logs.
