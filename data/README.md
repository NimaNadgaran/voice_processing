# `data/` — everything the app reads and writes

```
data/
├── raw/          uploads land here (one timestamped copy per job)
├── outputs/      one folder per job:  <job_id>/<path_id>/…
├── cache/        scratch (API uploads, NeMo work dirs)
└── datasets/     training/evaluation corpora, including explicitly requested downloads
```

Only this guide and the existing directory `.gitkeep` markers are committed.
Uploads, outputs, caches, datasets and local inventories such as
`installed-before.json` stay local. Runtime files are created as needed on a
fresh clone; do not publish private recordings or transcripts to GitHub.

---

## Output layout

```
data/outputs/job_20260901-013832_66b1de/
├── report.json                      ← the whole run: metrics, timings, comparison
├── meeting_original.wav             ← original audio decoded to WAV
├── path1/
│   ├── result.json
│   ├── meeting__path1__01_denoised_spectral_gate.wav
│   ├── meeting__path1__02_speaker01_diarize_cluster.wav
│   ├── meeting__path1__02_speaker02_diarize_cluster.wav
│   ├── meeting__path1__03_speaker01_faster_whisper.txt
│   ├── meeting__path1__03_speaker02_faster_whisper.txt
│   └── …
└── path4/
    └── …
```

`report.json` is the same object the frontend renders — safe to parse in your
own scripts:

```python
import json
report = json.load(open("data/outputs/<job>/report.json"))
report["speaker_estimate"]["n_speakers"]
report["paths"][0]["denoised"]["metrics"]["snr_improvement_db"]
report["comparison"]["ranking"]
```

Actual names include the sanitized source stem, path and method; prefixes may
be shortened for safe filenames. Text is plain UTF-8, including Persian.
TXT files appear when recognition succeeds or finds no speech; failures are
reported in `tracks[].transcription` while speaker audio remains available.
Per-path and whole-job ZIP downloads include all produced text/audio/reports.
See [API/report fields](../docs/API.md).

Old jobs are never deleted automatically. `data/outputs/` is just folders —
delete what you do not need.

---

## Datasets (training only)

Training speech is needed only for local model training. An example layout:

```
data/datasets/
├── clean/                  clean speech, any tree     (denoiser)
├── noise/                  noise recordings, any tree (denoiser, optional)
└── speakers/               ONE FOLDER PER SPEAKER     (separator)
    ├── speaker_0001/*.wav
    └── speaker_0002/*.wav
```

### Free corpora

| corpus | what | size | licence | link |
|---|---|---|---|---|
| **LibriSpeech** `train-clean-100` | 100 h clean English, 251 speakers | 6 GB | CC BY 4.0 | openslr.org/12 |
| **LibriSpeech** `train-clean-360` | 360 h, 921 speakers | 23 GB | CC BY 4.0 | openslr.org/12 |
| **VoiceBank (VCTK)** | 110 English speakers | 11 GB | ODC-By | datashare.ed.ac.uk |
| **Common Voice** | 100+ languages | varies | CC0 | commonvoice.mozilla.org |
| **VoxCeleb1** | 1 251 speakers, real-world | 39 GB | CC BY 4.0 | robots.ox.ac.uk/~vgg/data/voxceleb |
| **DEMAND** | 18 real noise environments | 4 GB | CC BY-SA | zenodo.org/records/1227121 |
| **MUSAN** | noise + music + babble | 11 GB | CC BY 4.0 | openslr.org/17 |
| **ESC-50** | 2 000 environmental clips | 600 MB | CC BY-NC | github.com/karolpiczak/ESC-50 |
| **WHAM!** | noise recorded in cafés/bars | 76 GB | CC BY-NC 4.0 | wham.whisper.ai |
| **FSDnoisy18k** | 20 h of varied noise | 10 GB | CC | zenodo.org/records/2529934 |

Check each licence before commercial use — several are non-commercial.

### Helper

```bash
python scripts/prepare_datasets.py --check                    # what have I got?
python scripts/prepare_datasets.py --links                    # download instructions
python scripts/prepare_datasets.py --organize-librispeech PATH --out data/datasets/speakers
python scripts/prepare_datasets.py --flat-to-speakers PATH --out data/datasets/speakers
```

The script never downloads multi-GB corpora behind your back — it prints the
commands and organises what you have already fetched.

Root **`python setup.py` is different**: explicitly running it prepares/trains
both local models and can download LibriSpeech `train-clean-100` (~6.4 GB) if
default training speech is missing. Even `--estimate-only` can prepare/download
data before measuring time. `--no-download` prevents that. Downloads and
archives remain ignored; see [training/ETA instructions](../docs/TRAINING.md).

Recorded-reference audits use a separately acquired `dev-clean` corpus or other
held-out references, not the training download. Keep evaluation speech out of
training and validate source licenses before sharing any corpus.

### How little can you get away with?

| goal | clean speech | noise | speakers |
|---|---|---|---|
| denoiser that beats `spectral_gate` on your mic | ~10 h | ~2 h (or synthetic) | — |
| denoiser that is genuinely good | ~100 h | ~20 h | — |
| separator that works at all | — | optional | ~50 speakers × 10 min |
| separator that is genuinely good | — | recommended | 250+ speakers |

The denoiser loader synthesises noise (white / pink / brown / mains hum /
impulsive) when `noise/` is empty, so you can start with speech alone.

---

## Privacy

Uploads stay on your machine. The only exceptions are the two `api_huggingface`
methods, which are disabled unless you set `HF_TOKEN` yourself and are labelled
"uploads audio" in the UI.

All four speech-to-text engines process voices locally after weight downloads.
The server has no authentication and should stay on localhost for private audio.
Old uploads/outputs are not auto-deleted; review exact folders before cleanup,
and do not erase training/checkpoint work to prepare a GitHub upload. Ignoring
files leaves them intact on disk. Completed job files persist across restarts,
but live in-process job state does not; see [API notes](../docs/API.md).
