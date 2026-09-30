# Denoise &amp; Separate

Upload a recording with several people in it -- **audio or video** (for a video
the audio track is extracted automatically). Get back:

1. **a denoised version** of the whole file, then
2. **one clean file per speaker** — 4 speakers in, 4 files out —

with the number of speakers, a confidence figure, timings and quality metrics
shown in a web UI, and **several denoiser × separator pipelines you can run side
by side and compare**.

```
                    ┌──────────────┐      ┌──────────────┐
  meeting.wav ─────►│  DENOISE     │─────►│  SEPARATE    │─────► speaker_01.wav
   (4 people,       │  9 backends  │  │   │  8 backends  │       speaker_02.wav
    noisy)          └──────────────┘  │   └──────────────┘       speaker_03.wav
                                      │                          speaker_04.wav
                                      └──► 01_denoised.wav  (output #1)
```

---

## Quick start

```bash
pip install -r requirements.txt
python run.py demo            # makes a synthetic 4-speaker test file
python run.py serve           # → http://127.0.0.1:8000
                              # port busy? it rolls to 8001, 8002, ...
```

`python run.py` with no arguments (or PyCharm's green Run button) also starts
the web UI and prints the list of other commands.

That is enough for a fully working app: two DSP denoisers and the clustering
separator need nothing but numpy and scipy. Then add the neural backends you
want:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install denoiser speechbrain           # the two that matter most
```

(`deepfilternet` is stronger still. Its pip package stops at CPython 3.11, so on
a newer interpreter grab the standalone binary instead — see the note below.)

Check what you have at any time:

```bash
python run.py doctor
```

> **Python version:** torch wheels lag new Python releases, and a few backends lag
> much further. On **Python 3.12+** the `deepfilternet` pip package cannot be
> installed at all — its Rust extension `DeepFilterLib` publishes wheels only up
> to CPython 3.11, its sdist is built with `pyo3 0.19` (which itself caps at
> CPython 3.12), and it pins `numpy<2`. Installing it would downgrade numpy and
> take torch down with it. **DeepFilterNet still works**: fetch the project's
> official standalone binary instead, which has the DFN3 weights compiled in and
> needs no Python at all:
>
> ```bash
> python scripts/download_models.py deepfilternet
> ```
>
> `clearvoice` and `resemble-enhance` have no such escape hatch on 3.12+; paths 2
> and 3 fall back to Demucs DNS64 and say so on the card. `run.py doctor` lists
> exactly what your interpreter can and cannot have.

---

## The project map

```
denoise_seprate/
├── run.py                    one CLI: serve / doctor / denoise / separate / pipeline / demo
├── src/
│   ├── core/                 shared: audio I/O, DSP, metrics, registry, dataclasses
│   ├── denoising/            ← STAGE 1  (README.md inside)
│   │   ├── methods/          9 backends, one file each
│   │   ├── architectures.py  SpectralUNet (the trainable one)
│   │   └── training/         train_unet.py + dataset + losses + config
│   ├── separation/           ← STAGE 2  (README.md inside)
│   │   ├── methods/          8 backends, one file each
│   │   ├── speaker_count.py  how many people are talking, + confidence
│   │   ├── chunking.py       long-file support with permutation stitching
│   │   ├── architectures.py  Conv-TasNet (the trainable one)
│   │   └── training/         train_convtasnet.py + dataset + PIT losses + config
│   ├── pipeline/             paths (denoiser+separator combos), runner, comparison
│   └── api/                  FastAPI server + in-process job manager
├── frontend/                 vanilla HTML/CSS/JS UI (README.md inside)
├── models/                   weights: yours in checkpoints/, downloads in pretrained/
├── data/                     raw uploads, outputs, cache, training datasets
├── scripts/                  make_demo_audio · download_models · prepare_datasets
│                             benchmark (accuracy vs truth) · selftest (end-to-end API)
└── tests/                    smoke tests that run without any optional dependency
```

**Read next:**
[`src/denoising/README.md`](src/denoising/README.md) ·
[`src/separation/README.md`](src/separation/README.md) ·
[`frontend/README.md`](frontend/README.md) ·
[`models/README.md`](models/README.md) ·
[`data/README.md`](data/README.md)

---

## Denoising — 9 ways

| key | method | needs | notes |
|---|---|---|---|
| `none` | bypass (control) | — | the A/B baseline |
| `spectral_gate` | spectral gating | — | built-in numpy gate; `noisereduce` is opt-in via `SPECTRAL_GATE_BACKEND` and measures worse |
| `wiener_mmse` | MMSE-LSA (Ephraim-Malah) | — | never invents artefacts |
| `rnnoise` | RNNoise (Xiph) | `pyrnnoise` | 85 kB GRU, no torch |
| `deepfilternet` | DeepFilterNet 3 | `deepfilternet`, or the standalone binary | **the default recommendation** — pip package needs Python ≤ 3.11; the binary works anywhere |
| `demucs_denoiser` | Demucs DNS64 | `denoiser` | best on clicks/slams |
| `demucs_vocals` | htdemucs vocal isolation | `demucs` | best when the background is music |
| `resemble_enhance` | diffusion restoration | `resemble-enhance` | repairs clipping; generative |
| `local_unet` | **your** SpectralUNet | train it | fully yours, offline |
| `api_huggingface` | HF Inference API | `HF_TOKEN` | ⚠ uploads your audio |

## Separation — 9 ways

| key | method | type | speakers | needs |
|---|---|---|---|---|
| `none` | bypass (single track) | — | 1 | — |
| `diarize_cluster` | clustering diarization | diarization | **any** | — (built in) |
| `pyannote` | pyannote.audio 3.1 | diarization | **any** | `pyannote.audio` + free token |
| `nemo_msdd` | NVIDIA NeMo MSDD | diarization | **any** | `nemo_toolkit[asr]` |
| `sepformer` | SepFormer | separation | 2–3 | `speechbrain` |
| `convtasnet_asteroid` | Conv-TasNet zoo | separation | 2–3 | `asteroid` |
| `mossformer_clearvoice` | MossFormer2 | separation | 2 | `clearvoice` |
| `local_convtasnet` | **your** Conv-TasNet | separation | as trained | train it |
| `api_huggingface` | HF Inference API | separation | 2–3 | ⚠ uploads your audio |

**Diarization vs separation matters** — a 4-speaker meeting needs diarization
(no open model outputs 4 sources). The separation README explains the choice in
one table; the UI just lets you run both and compare.

---

## Pipeline paths

A *path* is one denoiser + one separator. Tick as many as you like; they all run
on the same upload and the results are compared automatically.

| id | name | path | ready on a bare install |
|---|---|---|---|
| `path1` | Instant | `spectral_gate → diarize_cluster` | ✅ |
| `path2` | Balanced (recommended) | `deepfilternet → pyannote` | needs torch; falls back to `demucs_denoiser → diarize_cluster` |
| `path3` | Cocktail party | `deepfilternet → sepformer` | needs torch; falls back to `demucs_denoiser` |
| `path4` | Classic DSP | `wiener_mmse → diarize_cluster` | ✅ |
| `path5` | Music / TV background | `demucs_vocals → convtasnet_asteroid` | needs torch |
| `path6` | Your own models | `local_unet → local_convtasnet` | train them |
| `path7` | Cloud | `api_huggingface → api_huggingface` | needs `HF_TOKEN` |
| `path8` | Control | `none → diarize_cluster` | ✅ |
| `path9` | Real-time stack | `rnnoise → diarize_cluster` | needs `pyrnnoise` |

Under the preset cards the UI has a **row builder**: each row is one more path
— pick the denoising module (or *No denoising*) and the separation module (or
*No separation*), press **+** for another row, tick the rows you want to run.
Blocked paths and modules stay visible with the exact `pip install` line.

---

## Using it without the UI

```bash
# one file, one method
python run.py denoise  meeting.wav --method deepfilternet -o clean.wav
python run.py separate clean.wav   --method pyannote --speakers 4 -o speakers/

# the whole thing, several paths, with a comparison table
python run.py pipeline meeting.wav --paths path1,path4,path8 --speakers 4

# a custom combination
python run.py pipeline meeting.wav --denoiser rnnoise --separator sepformer
```

```python
from src.pipeline import PipelineOptions, resolve_path, run_pipeline

report = run_pipeline(
    "meeting.wav",
    [resolve_path({"id": "path1"}), resolve_path({"denoiser": "wiener_mmse",
                                                  "separator": "diarize_cluster"})],
    PipelineOptions(num_speakers=4),
    progress=lambda e: print(e["message"]),
)
print(report["comparison"]["ranking"])
```

Everything the UI shows is in `data/outputs/<job_id>/report.json`.

---

## What you get back

Per run:

* `<name>__<path>__01_denoised_<denoiser>.wav` — output #1, playable as soon as
  it is written
* `<name>__<path>__02_speakerNN_<separator>.wav` — one per speaker, aligned to
  the original timeline

  Every file carries the source name, the path that produced it and the method
  used, so outputs from different paths never collide when you download them
  into the same folder. Each file has its own player and download button, each
  path has a "download this path" zip, and there is a download-everything zip.
* **speaker count + confidence**, per-speaker talk time, turn count and a
  timeline of every turn
* **denoise metrics**: SNR gain, noise-floor drop, speech-preservation
  correlation, a 0-100 score
* **separation metrics**: cross-talk between tracks, energy conservation, score
* **timings** per stage and a × real-time factor
* fun stats: silence %, talk-share split, most talkative speaker, estimated
  overlap, a rough word count
* a **comparison** across paths with the winner of each metric starred

> All quality numbers are **reference-free estimates** — there is no clean
> ground truth for an arbitrary upload. They are honest for comparing runs on
> the same file; they are not PESQ. Both stage READMEs say exactly what each
> number means and where it lies to you.

---

## Training your own models

> **No training ever starts by itself.** The app and the CLI never import the
> training packages. You run these commands, or nothing trains.

```bash
pip install -r requirements-train.txt

# ALWAYS estimate first -- it times real steps on YOUR machine and then exits
python -m src.denoising.training.train_unet         --config src/denoising/training/config.yaml  --estimate-only
python -m src.separation.training.train_convtasnet  --config src/separation/training/config.yaml --estimate-only

# then, when you are happy with the ETA
python -m src.denoising.training.train_unet        --config src/denoising/training/config.yaml
python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml
```

Rough full-run times (details, per-epoch numbers and quality expectations are in
the stage READMEs — and the scripts print a measured ETA for your machine):

| model | preset | CPU (4 cores) | RTX 3060 / T4 | A100 / 4090 |
|---|---|---|---|---|
| SpectralUNet denoiser | `tiny` | ~4 h | ~15 min | ~5 min |
| SpectralUNet denoiser | `base` | ~7 days ⚠ | ~7 h | ~2 h |
| Conv-TasNet separator | `tiny` | ~13 h | ~40 min | ~15 min |
| Conv-TasNet separator | `base` | ~9 days ⚠ | ~10 h | ~3 h |

Both scripts support `--max-hours N` (stop cleanly, keep the checkpoint) and
`--resume`. Denoisers reach ~80 % of their final quality in the first 10–15
epochs; separators need far longer.

---

## Privacy

Everything runs locally. The two `api_huggingface` methods are the only
exception — they upload your audio to Hugging Face, they are **disabled unless
you set `HF_TOKEN` yourself**, and the UI labels them "uploads audio".

---

## Performance notes

* BLAS threads are capped (`DS_THREADS`, default `min(4, cores)`) so a run does
  not take over a busy machine. Raise it if you want more speed.
* Paths run sequentially inside a job; two jobs run concurrently
  (`JobManager(workers=2)`).
* Long files are chunked automatically — 10 s blocks for transformer separators
  with permutation stitching so speakers never swap files mid-recording.
* Models are loaded once and cached in the registry across jobs.

## Input formats

| kind | formats |
|---|---|
| audio | wav, flac, ogg, opus, mp3, m4a, aac, wma, aiff, caf, amr, au |
| **video** | mp4, m4v, mov, mkv, webm, avi, wmv, flv, mpg, mpeg, ts, 3gp, ogv … |

For a video only the audio stream is decoded (`-vn`), so a 2 GB screen recording
costs about as much to read as its soundtrack alone. Uploads stream to disk in
1 MB chunks, so file size costs disk rather than RAM (limit `MAX_UPLOAD_MB`,
default 1024).

Video needs ffmpeg. `pip install -r requirements.txt` includes
**imageio-ffmpeg**, which bundles a static binary and needs no admin rights; a
system ffmpeg on `PATH` is used in preference if you have one. `python run.py
doctor` tells you which was found.

## Testing and measuring

```bash
python tests/test_smoke.py                       # 23 unit tests, no optional deps
python scripts/selftest.py --url http://127.0.0.1:8000   # 59 end-to-end API checks
python scripts/benchmark.py --source synthetic   # accuracy against known truth
```

* **test_smoke.py** — audio I/O, DSP round-trips, metrics, the registry, both
  always-available backends, the speaker counter, a full two-path pipeline.
* **selftest.py** — drives the *running server*: every endpoint, uploads real
  audio and a real video, checks the produced wavs decode, verifies the zips,
  the unique file naming, and that failures come back as sentences rather than
  tracebacks.
* **benchmark.py** — builds test mixtures whose ground truth it knows, so it can
  report real accuracy: speaker-count exactness, coverage, purity, SI-SDRi and
  true denoising gain. This is how the defaults in this project were chosen —
  see the note on MMSE-LSA tuning in the denoising README.

## Error handling

A Python traceback never reaches the browser. Every failure is turned into a
sentence plus, where one exists, a concrete fix ("install ffmpeg", "pick a
shorter clip"). The full traceback still goes to the server console and to
`data/cache/errors.log`, so nothing is lost for debugging.

## Licence

The code here is yours to use. Each optional backend keeps its own licence
(MIT / Apache-2.0 / CC BY-NC for some datasets) — `models/README.md` lists them.
