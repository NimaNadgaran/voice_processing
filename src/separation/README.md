# Speaker separation

Suggested algorithm: **built-in Clustering Diarization (`diarize_cluster`)**
for meetings/interviews with mostly turn-taking speakers. Pair it with
DeepFilterNet (`deepfilternet`) for the suggested denoise/separate workflow.
Clustering requires no gated-model account, but cannot acoustically separate
simultaneous voices. Optional ECAPA embeddings download if SpeechBrain is used.
For sustained overlap, compare the fixed-source neural separators instead.

Latest adapter fixes, real-audio results and access limits:
[AUDIO_VALIDATION.md](../../AUDIO_VALIDATION.md). Local model setup/training is
available through `python setup.py` at the project root.

> Stage 2 of the pipeline. Input: one (denoised) recording with several people.
> Output: **one file per speaker**, all aligned to the original timeline.
> Eight interchangeable backends, plus a speaker counter.

```python
from src.separation import count_speakers, separate

print(count_speakers("meeting.wav"))          # {'n_speakers': 4, 'confidence': 0.82, …}
res = separate("meeting.wav", method="diarize_cluster", output_dir="out/")
for track in res.tracks:
    print(track.label, track.total_speech, track.path)
```

---

## 1. The distinction that decides everything: diarization vs source separation

These are two genuinely different problems and the right choice depends on your
recording. Getting this wrong is the single most common reason people are
disappointed by their results.

|  | **Diarization** ("who spoke when") | **Source separation** ("unmix the voices") |
|---|---|---|
| what it does | cuts the timeline into speaker turns | acoustically separates simultaneous voices |
| output for speaker *i* | their turns, silence elsewhere | their voice, continuously, even under others |
| number of speakers | **any** — discovered from the data | **fixed** by the checkpoint (2 or 3) |
| overlapping speech | both voices land in both files | genuinely split |
| cost | cheap | expensive |
| in this project | `diarize_cluster`, `pyannote`, `nemo_msdd` | `sepformer`, `convtasnet_asteroid`, `mossformer_clearvoice`, `local_convtasnet` |

**Rule of thumb**

```
Meeting / interview / podcast — people mostly take turns   → built-in clustering (diarize_cluster)
Two people talking over each other the whole time          → separation   (sepformer, mossformer)
4+ people AND heavy overlap                                → diarization first; separation on the overlapped spans
Don't know                                                 → run both as two paths and compare in the UI
```

A 4-speaker meeting needs diarization with the supplied pretrained checkpoints,
which output at most three sources. `diarize_cluster` is the built-in default —
it is the only family that actually answers "give me 4 files for 4 people".

---

## 2. The nine backends

| key | name | type | speakers | speed | quality | offline | install |
|---|---|---|---|---|---|---|---|
| `none` | No separation (bypass) | — | 1 | instant | ●○○○○ | ✅ | built in |
| `diarize_cluster` | Clustering diarization | diarization | **any** | fast | ●●●○○ | ✅ | built in |
| `pyannote` | pyannote.audio 3.1 | diarization | **any** | medium | ●●●●● | ✅ | `pip install pyannote.audio` + free token |
| `nemo_msdd` | NVIDIA NeMo MSDD | diarization | **any** | slow | ●●●●● | ✅ | `pip install "nemo_toolkit[asr]"` |
| `sepformer` | SepFormer (SpeechBrain) | separation | 2–3 | slow | ●●●●● | ✅ | `pip install speechbrain` |
| `convtasnet_asteroid` | Conv-TasNet (Asteroid zoo) | separation | 2–3 | medium | ●●●●○ | ✅ | `pip install asteroid` |
| `mossformer_clearvoice` | MossFormer2 (ClearerVoice) | separation | 2 | slow | ●●●●● | ✅ | `pip install clearvoice` |
| `local_convtasnet` | **Your** Conv-TasNet | separation | as trained | fast | ●●●●○ | ✅ | train it (§6) |
| `api_huggingface` | Hugging Face API | separation | 2–3 | medium | ●●●●○ | ❌ **uploads audio** | free HF token |

Everything except `api_huggingface` runs locally. `api_huggingface` stays
disabled until you set `HF_TOKEN` and configure a deployed compatible audio
endpoint with `HF_SEPARATION_URL` or a hosted `HF_SEPARATION_MODEL`. A model
existing on the Hub does not imply it has hosted inference.

### Notes per backend

**`diarize_cluster`** — the always-available one, pure numpy:

```
audio ─► VAD ─► 1.5 s windows ─► embeddings ─► estimate K ─► 0.6 s windows
      ─► assign to K centroids ─► median-filter labels in time
      ─► merge into turns ─► one cosine-faded mask per speaker ─► N files
```

It automatically uses **SpeechBrain ECAPA-TDNN** embeddings if `speechbrain` is
installed (much better) and falls back to MFCC-based features otherwise. The
metrics report which one ran (`"embedding": "ecapa-tdnn" | "mfcc-numpy"`).

**`pyannote`** — an optional neural diarization alternative. Neural segmentation *with overlap
detection* + agglomerative clustering of embeddings. Finds the speaker count on
its own, accepts a pinned count, and reports real turn boundaries. Requires
model access and a read token: 4.x uses `speaker-diarization-community-1`, while
3.x uses `speaker-diarization-3.1` / `segmentation-3.0`. Accept the conditions
for the version actually installed; a token alone does not establish access.

**`sepformer`** — dual-path transformer, permutation-invariant SI-SNR training.
Genuinely unmixes simultaneous speech. Quadratic in length, so long files are
processed in overlapping 10 s blocks with **permutation stitching** (see §4).
Auto-selects the 2-source or 3-source checkpoint; override with
`SEPFORMER_MODEL`.

**`convtasnet_asteroid`** — 3–5× faster than SepFormer with most of the quality.
Override with `ASTEROID_MODEL`.

**`nemo_msdd`** — multi-scale diarization decoder; fuses embeddings at several
window scales and decodes overlap pairwise. Marked *experimental* here only
because NeMo's API has moved between releases — the wrapper probes two call
styles and falls back to parsing the RTTM it writes.

---

## 3. Counting the speakers

`src/separation/speaker_count.py`. This runs before separation and its answer is
passed to the separator.

1. **VAD** → keep speech only.
2. Slice into 1.5 s windows (0.75 s hop), keep windows that are ≥ 60 % speech.
3. **Embed** each window:
   * ECAPA-TDNN (192-d) when SpeechBrain is installed, else
   * a numpy fallback: 12 MFCC means + 7 MFCC stds + **pitch (×3 weight)** +
     spectral slope + centroid, all CMVN-normalised.
4. Cluster for every K in 2…max and score with the **silhouette coefficient**;
   independently compute the **eigengap** of the normalised Laplacian.
5. Pick the best-scoring K. If the best silhouette is below 0.14, call it one
   speaker.

`confidence` is the normalised margin between the best K and the runner-up, plus
a bonus when the eigengap agrees. **It is a measure of how clearly the data
preferred that K — not an accuracy guarantee.** Measured accuracy with the numpy
fallback is 17–33% exact on hard material — see §5b. Pin the count when you know
it, and install `speechbrain` when you do not.

### Two design details that matter

**Pitch is weighted ×3.** It is by far the most speaker-discriminative scalar
available, and without the weight it is drowned out by the 19 cepstral
dimensions. Measured on the bundled 4-speaker demo, that one change moved the
answer from 2 speakers to the correct 4.

**`c0` (overall loudness) is dropped.** It tracks how close someone sat to the
microphone, not who they are — keeping it makes clusters split on volume.

### Counting runs on the *original*, not the denoised audio

This is the pipeline's default (`PipelineOptions.count_on="original"`) and it is
deliberate. Measured on the bundled demo (4 speakers, 7 dB SNR):

| counted on | speakers found | speakers recovered when the count is pinned to 4 |
|---|---|---|
| original audio | **4 ✅** | 4/4 |
| after aggressive spectral gating | 2 ❌ | 4/4 |

Strong denoising strips exactly the low-level spectral detail that speaker
embeddings rely on. Separation still benefits from the cleaner signal, so the
pipeline **counts on the raw audio and separates on the clean audio**. Switch it
with `count_on: "denoised"` if your denoiser is gentle.

**If you know the number, pin it.** A pinned count always beats an estimate;
the UI has a dropdown for exactly this.

---

## 4. Long files: permutation stitching

Fixed-length separators cannot swallow a 20-minute meeting, so
`src/separation/chunking.py` cuts the signal into overlapping blocks.

The subtle part: a separator has no notion of identity, so block 1 may output
(Alice, Bob) while block 2 outputs (Bob, Alice). Before overlap-adding a block
we test every permutation against the tail of the already-assembled output and
keep the one with the highest correlation in the overlap region.

```
block 1:  ┌── Alice ──┐          keep as is
          └── Bob ────┘
block 2:      ┌── Bob ────┐      correlate the overlap → swap
              └── Alice ──┘
result:   ┌── Alice ──────────┐  one speaker per file, all the way through
          └── Bob ────────────┘
```

Without this step the speakers swap files every few seconds. It is the single
most important piece of glue for using these models on real recordings.

---

## 5. Reading the metrics

Reference-free again — there is no ground truth for an arbitrary upload.

| metric | meaning | good value |
|---|---|---|
| `n_speakers` | tracks written | should match reality |
| `separation_score` | 70 % (1 − cross-talk) + 30 % energy conservation | > 70 |
| `mean_cross_correlation` | average \|correlation\| between output tracks | **< 0.2**; > 0.5 means the tracks are near-copies |
| `energy_conservation` | 1 − ‖mix − Σ sources‖² / ‖mix‖² | close to 1.0 |
| per track: `total_speech` | seconds of detected speech | — |
| per track: `segments` | `[[start, end], …]` turn boundaries | drives the UI timeline |
| per track: `speech_ratio` | fraction of the track that is speech | a very low value = a nearly empty stem |

Empty and duplicate stems are dropped automatically: a source more than 32 dB
below the loudest, or with under 0.35 s of speech, is discarded (that is what a
2-source model emits when the file has one speaker).

### With ground truth

```python
from src.core.metrics import permutation_si_sdr
permutation_si_sdr([est1, est2], [ref1, ref2])   # best-permutation SI-SDR
```

`scripts/make_demo_audio.py` writes per-speaker ground truth next to the mixture,
so you can score any backend honestly without downloading a corpus.

---

## 5b. How accurate is it, really? (measured, not claimed)

`scripts/benchmark.py` builds mixtures whose ground truth it knows, so these are
real numbers rather than the reference-free proxies the UI shows.

**Test set:** LibriSpeech `dev-clean` (real recorded speech, 40 speakers),
24 s conversations, 2/3/4 speakers at 5 dB and 15 dB SNR, with dense turn-taking
and occasional overlap. Only the bare-install paths could run (no PyTorch on
this machine), so this measures the **numpy fallback**, which is the weakest
configuration on purpose.

| path | denoise gain | speaker count exact | coverage | purity | SI-SDRi |
|---|---|---|---|---|---|
| `path1` spectral_gate -> cluster | **+2.2 dB** | 17-33% | 0.72 | 0.64 | -0.7 dB |
| `path4` wiener_mmse -> cluster | **+5.0 dB** | 17-33% | 0.82 | 0.71 | +0.7 dB |
| `path8` none -> cluster (control) | 0.0 dB | 17-33% | 0.76 | 0.67 | +0.1 dB |

Read honestly, that says:

* **Denoising works.** MMSE-LSA gains +5 dB SI-SDR against the true clean
  mixture, and denoising before separating measurably helps (`path4` beats the
  no-denoise control on every separation metric).
* **The numpy fallback separator is a baseline, not a solution.** Coverage
  around 0.7-0.8 means roughly three of four speakers get their own file;
  purity around 0.7 means each file still carries some of someone else.
* **Speaker counting with MFCC features is weak** on this material -- 17-33%
  exactly right, typically off by one. Thirteen different selection rules were
  compared (silhouette variants, cosine space, PCA, penalised silhouette,
  elbow, eigengap, agglomerative thresholds) and they all landed in the same
  10-30% band, so this is not a tuning problem: 20-dimensional cepstral
  statistics simply are not discriminative enough on 24 s of noisy,
  continuously-overlapping speech.

### What actually fixes it

1. **Pin the count** if you know it. The UI dropdown does this and it removes
   the counting error entirely.
2. **`pip install speechbrain`.** Both the counter and `diarize_cluster`
   automatically switch to ECAPA-TDNN speaker embeddings, which are *trained*
   to be speaker-discriminative rather than being generic spectral statistics.
   This is the single biggest quality lever in the project. (Not measured here
   -- this machine has no PyTorch. The published gap between cepstral features
   and ECAPA on diarization tasks is large, but treat that as literature, not
   as a number from this benchmark.)
3. **Use `pyannote`** for real meetings, which additionally detects overlap.

### Reproduce it

```bash
python scripts/prepare_datasets.py --links          # get LibriSpeech dev-clean
python scripts/benchmark.py --source librispeech \
    --corpus data/datasets/_download/LibriSpeech/dev-clean \
    --speakers 2,3,4 --snr 5,15

python scripts/benchmark.py --source librispeech --pin-speakers   # isolate separation
python scripts/benchmark.py --source synthetic                    # no download needed
```

### One measurement that changed the code

The VAD used a threshold anchored to the clip's 10th-percentile energy. On
recordings that are 97-99% speech -- normal in an interview, universal in
concatenated corpus audio -- that "floor" *is* speech, so the threshold sat
inside the voice and discarded half of it. One 4-speaker case produced **zero**
usable windows, which made speaker counting impossible before it started.

The fix keeps the relative threshold but adds a spectral test: speech is
harmonic (low flatness) *and* puts most of its energy between 200 Hz and 4 kHz,
while hiss is broadband and mains hum has nothing in that band. Frames that pass
both tests are kept even when they sit near the floor. Usable windows in the
worst case went from 2 to 12, and hum/hiss/silence are still correctly rejected.

---

## 6. Training the local model

> **Nothing trains automatically.** The app never starts a training run.

### 6.1 Architecture (`src/separation/architectures.py`)

Conv-TasNet (Luo & Mesgarani, 2019):

```
mixture ─► Encoder (1-D conv, stride L/2) ─► w  ─────────────────┐
                                            │                    │
                                    Separator (TCN: R repeats    │
                                    × X dilated depthwise blocks)│
                                            │                    │
                                       masks m₁…m_C              │
                                            ▼                    ▼
                             source_i = Decoder(w · m_i)  ◄──────┘
```

Trained with **utterance-level permutation-invariant SI-SNR (uPIT)** — score
every assignment of outputs to references and back-propagate only the best one.
Without PIT the model gets contradictory gradients ("output 1 should be Alice"
then "output 1 should be Bob") and collapses to emitting the mixture twice.

Optional extra term: **mixture consistency** (the sources should sum back to the
mixture), weight 0.1. It measurably reduces the "energy leaks into nowhere"
failure mode on real recordings.

| preset | params | N/B/H/X/R | sample rate | crop |
|---|---|---|---|---|
| `tiny` | ~0.9 M | 128/64/128/5/2 | 8 kHz | 2 s |
| `base` | ~3.5 M | 256/128/256/7/2 | 8 kHz | 4 s |
| `paper` | ~5.1 M | 512/128/512/8/3 | 8 kHz | 4 s |

### 6.2 Data — one folder per speaker

```
data/datasets/speakers/
├── speaker_0001/  *.wav
├── speaker_0002/  *.wav
└── …
```

LibriSpeech's `<speaker>/<chapter>/*.flac` layout is auto-detected. A flat
folder falls back to "one speaker per file" **with a printed warning** — that
still works but is much weaker.

Mixtures are built on the fly, WSJ0-mix style: pick `n_src` *different*
speakers, take an active (non-silent) crop from each, scale to a random relative
level (±2.5 dB), sum. Mixing two clips of the *same* person teaches nothing —
the permutation is ambiguous — which is why speaker grouping is enforced.

Set `noise_dir` to also add background noise (the WHAM! recipe). A noisy-trained
separator transfers far better to real recordings.

| purpose | corpus | size | note |
|---|---|---|---|
| speakers | LibriSpeech `train-clean-100` | 6 GB | 251 speakers — the standard |
| speakers | LibriSpeech `train-clean-360` | 23 GB | 921 speakers, clearly better |
| speakers | VoxCeleb1 | 39 GB | 1 251 speakers, real-world audio |
| speakers | Common Voice | varies | best for non-English |
| noise | WHAM! / MUSAN / DEMAND | — | see the denoising README |

**Minimum that works:** ~50 speakers × ~10 minutes each. Below ~20 speakers the
model memorises voices instead of learning to separate them (the script warns).

### 6.3 Commands

```bash
pip install -r requirements-train.txt

# 1) ALWAYS first: times real steps on YOUR machine, prints the true ETA, exits.
python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml --estimate-only

# 2) for real
python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml

# CPU-friendly
python -m src.separation.training.train_convtasnet --preset tiny --clean-dir data/datasets/speakers

# three speakers
python -m src.separation.training.train_convtasnet --config … --n-src 3

# time-boxed / resumable
python -m src.separation.training.train_convtasnet --config … --max-hours 8
python -m src.separation.training.train_convtasnet --config … --resume models/checkpoints/separation_convtasnet_last.pt
```

Outputs `models/checkpoints/separation_convtasnet_{last,best}.pt`. As soon as
`…_best.pt` exists, **Local Conv-TasNet (your model)** turns green in the UI.

### 6.4 How long does training take?

Full runs, dataset on a local SSD. **The script measures your machine and prints
a real ETA before starting — trust that over this table.**

| preset | steps/epoch × epochs | CPU (4 cores) | RTX 3060 / T4 | A100 / 4090 |
|---|---|---|---|---|
| `tiny` (0.9 M, 8 kHz, 2 s, bs 8) | 400 × 40 = 16 k | ≈ 3 s/step → **~13 hours** | ≈ 0.15 s/step → **~40 min** | **~15 min** |
| `base` (3.5 M, 8 kHz, 4 s, bs 4) | 1 000 × 100 = 100 k | ≈ 8 s/step → **~9 days** ⚠ not practical | ≈ 0.35 s/step → **~10 hours** | **~3 hours** |
| `paper` (5.1 M, 8 kHz, 4 s, bs 4) | 2 000 × 200 = 400 k | ⚠ not practical | ≈ 0.8 s/step → **~3.7 days** | **~1 day** |

Per epoch:

| preset | CPU | RTX 3060 | A100 |
|---|---|---|---|
| `tiny` | ~20 min | ~1 min | ~22 s |
| `base` | ~2.2 h | ~6 min | ~1.8 min |
| `paper` | ~4.5 h | ~27 min | ~7 min |

**Separation converges much more slowly than denoising.** Expect roughly:

| after | `base` on a 3060 | typical val SI-SDRi |
|---|---|---|
| 5 epochs (~30 min) | the model has found "two voices exist" | +4 to +6 dB |
| 20 epochs (~2 h) | usable | +8 to +10 dB |
| 100 epochs (~10 h) | converged | +11 to +13 dB |
| `paper`, 200 epochs on a good GPU | reference quality | +14 to +15 dB (WSJ0-2mix) |

For reference: published Conv-TasNet reaches 15.3 dB SI-SDRi on WSJ0-2mix, and
SepFormer ~22 dB. A locally trained model will not beat those *in general* —
but on your speakers, your language and your recording chain it very well can.

### 6.5 Tuning notes

* **Loss stuck around −6 dB and both outputs sound like the mixture** — classic
  PIT collapse. Lower the LR to 5e-4, check that mixtures really come from
  *different* speakers, and confirm the crops are not silence.
* **`n_src` is baked into the checkpoint.** A 2-source model cannot output 3.
  Train a second checkpoint and point `SEPARATION_CKPT` at it.
* **`causal: true` + `norm: cLN`** gives a streaming-capable model (~2 dB worse).
* **8 kHz is not a compromise** — it is the WSJ0-mix standard, and 4× cheaper
  than 16 kHz. The pipeline resamples back to your original rate.
* **`segment_seconds`** — longer crops help the TCN's receptive field but cost
  linearly. 4 s is the sweet spot.
* **Mixture consistency** — set to 0 if you are reproducing paper numbers; keep
  0.1 for real-world audio.

---

## 7. Adding your own backend

```python
# src/separation/methods/my_separator.py
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ..base import BaseSeparator

@register_separator
class MySeparator(BaseSeparator):
    info = MethodInfo(
        key="my_sep", name="My Separator", kind="separate", family="dsp",
        description="…", max_speakers=None, pip=["some_package"],
    )
    target_sr = 16000

    def _separate(self, audio: AudioBuffer, num_speakers):
        sources = [...]                       # list of numpy float32 arrays
        return sources, {
            "backend": "mine",
            "output_sr": 16000,               # if you resampled internally
            "confidence": 0.8,
            "labels": ["Speaker 1", "Speaker 2"],
            "segments": [[[0.0, 4.2]], [[4.2, 9.0]]],   # optional
            "metrics": {"anything": "you like"},
        }
```

Add the module name to `MODULES` in `methods/__init__.py`. The base class
resamples, drops empty/duplicate stems, runs VAD per track, computes turn
boundaries, scores the separation and captures errors.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| all neural separators greyed out | no torch | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| `pyannote` says "accept the licence" | model gate not accepted | visit the two model pages on huggingface.co, accept, set `HF_TOKEN` |
| got 2 files but there are 4 people | a 2-source model was used | use `pyannote` or `diarize_cluster` — see §1 |
| speakers swap between files | (fixed here) permutation stitching | ensure you are on `chunked_separate`, not calling the model directly |
| one output is silence | model emitted an empty stem | already dropped automatically; if not, lower `silence_floor_db` |
| everyone lands in one file | count estimated as 1 | pin the count in the UI, or install `speechbrain` for ECAPA embeddings |
| overlapping speech is in both files | that is diarization behaving correctly | use `sepformer` / `mossformer_clearvoice` |
| `local_convtasnet` unavailable | not trained yet | §6 |
| very slow on a long file | transformer separators are quadratic | use `convtasnet_asteroid` or `diarize_cluster` |

---

## 9. References

- Luo & Mesgarani (2019), *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking*, IEEE/ACM TASLP.
- Kolbæk et al. (2017), *Multitalker Speech Separation with Utterance-level PIT*, IEEE/ACM TASLP.
- Subakan et al. (2021), *Attention is All You Need in Speech Separation* (SepFormer), ICASSP.
- Zhao et al. (2023), *MossFormer / MossFormer2*, ICASSP.
- Bredin et al. (2023), *pyannote.audio 2.1 / 3.1 speaker diarization pipeline*, INTERSPEECH.
- Park et al. (2022), *Multi-scale Speaker Diarization with Dynamic Scale Weighting* (MSDD), INTERSPEECH.
- Desplanques et al. (2020), *ECAPA-TDNN*, INTERSPEECH.
- Wichern et al. (2019), *WHAM!: Extending Speech Separation to Noisy Environments*, INTERSPEECH.

**Previous stage:** [`../denoising/README.md`](../denoising/README.md)
