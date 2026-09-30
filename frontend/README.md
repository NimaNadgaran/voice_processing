# `frontend/` — the web UI

Vanilla HTML + CSS + JS. **No build step, no npm, no framework.** FastAPI serves
`index.html` at `/` and this folder at `/static`, so editing a file and hitting
refresh is the whole development loop.

```
frontend/
├── index.html     structure (upload → paths → options → run → results → compare)
├── styles.css     design tokens + every component; dark by default, light theme included
├── app.js         all behaviour (~700 lines, no dependencies)
└── assets/
    └── demo_conversation.wav    bundled 4-speaker demo for the "try it" button
```

---

## What the user goes through

```
1  drop a file          → decoded in the browser, waveform drawn on a canvas
2  tick pipeline paths  → 9 preset cards + a row builder (+ adds another path)
3  options              → pin the speaker count, choose what to count on, normalise
4  run                  → POST /api/jobs, then poll
5  results appear ONE BY ONE as the backend writes each file
6  compare              → ranking bars + a metric table with the winner starred
```

### Results really do stream in

The backend emits an `artifact` event the moment each wav hits disk. `app.js`
turns those into player cards immediately, so the denoised file is playable
while the separator is still running. When the job finishes, the partial cards
are replaced by the full render (metrics, waveforms, timelines, fun stats).

That is why `handleArtifact()` and `renderResultCard()` both exist — one is the
optimistic incremental view, the other the authoritative final one.

---

## API contract used

| call | purpose |
|---|---|
| `GET /api/health` | the status pill in the top bar |
| `GET /api/methods` | method browser + the row builder's dropdowns |
| `GET /api/paths` | preset cards, including availability and install hints |
| `POST /api/jobs` | multipart: `file`, `paths` (JSON array), `options` (JSON object) |
| `GET /api/jobs/{id}?since=N` | poll: status, `pct`, new events, and the report when done |
| `GET /api/files/{job}/{path}/{file}` | audio for the `<audio>` players and download links |
| `GET /api/jobs/{id}/zip` | "download everything" |

Polling every 700 ms was chosen over SSE/WebSockets on purpose: it survives
reloads, proxies and sleep/wake, and a job that takes 10 s does not need a
socket. Events carry a `seq`, so `?since=` only ever returns what you have not
seen.

---

## Design notes

* **Tokens in `:root`** — every colour, radius and shadow. The light theme
  overrides the same names under `html[data-theme="light"]`; the toggle stores
  the choice in `localStorage`.
* **Canvas waveforms** — peak arrays (500 points) come from the backend for
  results and from `decodeAudioData` for the local preview. Redrawn on resize
  and on theme change (`data-peaks` is stashed on each canvas).
* **Speaker colours** are a fixed 8-colour array indexed by speaker number, so
  the swatch, the talk-share bar and the timeline always agree.
* **Timelines** are absolutely-positioned divs, one per turn, `left`/`width` in
  percent of the clip duration — so every speaker's timeline lines up with the
  others and with the original audio.
* **Blocked paths** stay visible and greyed out with the exact `pip install`
  command, instead of being hidden. Discoverability beats tidiness here.
* **Selections persist** in `localStorage` (`ds-selected`, `ds-custom`) and are
  re-validated against the server's availability on load.

---

## Customising

| want to… | do this |
|---|---|
| change the colours | edit the `:root` block in `styles.css` |
| add a preset path card | add it to `PRESET_PATHS` in `src/pipeline/paths.py` — the UI picks it up automatically |
| show another metric | add it to `metric(...)` in `renderResultCard()` (`app.js`) |
| add a comparison row | add it to `METRIC_SPECS` in `src/pipeline/comparison.py` |
| change the poll rate | `POLL_MS` at the top of `app.js` |
| raise the upload limit | `MAX_UPLOAD_MB` env var (server side) |
| replace the demo clip | overwrite `assets/demo_conversation.wav`, or regenerate with `python run.py demo` |

## Speech-to-text controls and downloads

Options includes a recognizer and language selector. Auto chooses an installed
multilingual engine; Persian-specialized Wav2Vec2 and English/Persian Vosk have
language restrictions shown in the UI. Custom paths can inherit the global
recognizer, disable transcription, or select a different engine.

Every successfully transcribed speaker row has a safe plain-text preview
(`dir="auto"` for Persian/Arabic) and **Download text file** link. UTF-8 text
downloads and all ZIP archives include transcripts. Recognition failure shows
its reason without removing the speaker's audio player/download. No
transcription shows audio only. Model and language preferences persist in
`localStorage` (`ds-stt-method`, `ds-stt-language`).

`GET /api/methods` supplies recognizers and language metadata; the existing
`GET /api/files/{job}/{path}/{file}` endpoint also serves text files. See
[Stage 3 documentation](../src/transcription/README.md).

```bash
node tests/test_frontend_transcription.cjs
# Optional real-browser check, with playwright and an installed Chrome:
python scripts/check_transcription_ui.py --url http://127.0.0.1:8000
```

## Browser support

Anything current (Chrome, Edge, Firefox, Safari). Uses `fetch`, `AudioContext`,
CSS grid, `color-mix()` and `backdrop-filter`. No polyfills, no IE.
