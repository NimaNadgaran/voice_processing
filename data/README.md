# `data/` — everything the app reads and writes

```
data/
├── raw/          uploads land here (one timestamped copy per job)
├── outputs/      one folder per job:  <job_id>/<path_id>/…
├── cache/        scratch (API uploads, NeMo work dirs)
└── datasets/     TRAINING corpora — you put these here, nothing downloads by itself
```

Nothing in here is committed to git.

---

## Output layout

```
data/outputs/job_20260901-013832_66b1de/
├── report.json                      ← the whole run: metrics, timings, comparison
├── original_meeting.wav             ← exactly what you uploaded (decoded to wav)
├── path1/
│   ├── result.json
│   ├── 01_denoised.wav              ← first output
│   ├── 02_speaker_01.wav            ← one file per speaker
│   ├── 02_speaker_02.wav
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

Old jobs are never deleted automatically. `data/outputs/` is just folders —
delete what you do not need.

---

## Datasets (training only)

Only needed if you train the local models. Expected layout:

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
