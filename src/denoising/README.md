# Denoising

Suggested algorithm: **DeepFilterNet 3 (`deepfilternet`)**, followed by
built-in clustering (`diarize_cluster`) when speakers mostly take turns.
For modern Python environments, fetch the standalone denoiser with
`python scripts/download_models.py deepfilternet`; no DeepFilterNet pip package
or gated-model account is required for that route. If there is no compatible
standalone release for your platform, the backend reports it explicitly.
Compare a no-denoising control on overlapping speech; suppression is not
guaranteed to preserve every simultaneous voice.

Latest tuning, measured results, dependency limits and reproducible checks:
[AUDIO_VALIDATION.md](../../AUDIO_VALIDATION.md). Both local models can be
prepared and trained sequentially with `python setup.py` from the project root.

> Stage 1 of the pipeline. Input: one noisy recording. Output: one clean file.
> Nine processing methods plus bypass, behind the same three-line interface.

```python
from src.denoising import denoise

result = denoise("meeting.wav", method="deepfilternet", output_path="clean.wav")
print(result.metrics["snr_improvement_db"])   # e.g. 11.4
```

---

## 1. What this stage is actually doing

Almost every denoiser here — classic or neural — does the same thing under the
hood: it estimates a **mask** and multiplies the noisy signal by it.

```
noisy audio ──► analysis ──► |X(f,t)|  ──► [ estimate mask M(f,t) ∈ [0,1] ] ──┐
                                                                              │
clean audio ◄── synthesis ◄── M(f,t) · X(f,t) ◄───────────────────────────────┘
```

What separates the methods is **how the mask is estimated**:

| family | how the mask is decided | learns from |
|---|---|---|
| spectral gating | "is this bin louder than the noise profile?" | the quiet parts of *your* clip |
| MMSE‑LSA / Wiener | statistical estimator of the clean amplitude | a running noise model |
| RNNoise | a 85 kB GRU predicts 22 band gains | ~100 h of speech+noise |
| DeepFilterNet | ERB gains **+ complex deep filters** | ~500 h (DNS corpus) |
| Demucs / denoiser | no mask at all — a U‑Net regenerates the waveform | DNS challenge data |
| Demucs vocals | separates music stems, keeps the vocal one | MUSDB + internal |
| Resemble Enhance | a diffusion model *regenerates* clean speech | large TTS-grade corpora |
| your local U‑Net | whatever **you** train it on | your data |

The practical consequence: mask-based methods primarily attenuate what is
there and cannot restore a destroyed band. They can still distort speech or
produce filtering artifacts; keep the original when preservation matters.
Generative methods (Resemble Enhance) can restore, but may alter timbre —
which is why the UI flags it.

---

## 2. Nine methods plus bypass

| key | name | family | speed | quality | offline | install |
|---|---|---|---|---|---|---|
| `none` | No denoising (control) | dsp | instant | — | ✅ | built in |
| `spectral_gate` | Spectral Gating | dsp | realtime | ●●●○○ | ✅ | built in (`noisereduce` is opt-in and measures worse) |
| `wiener_mmse` | MMSE‑LSA (Ephraim‑Malah) | dsp | realtime | ●●●○○ | ✅ | built in |
| `rnnoise` | RNNoise (Xiph) | pretrained | realtime | ●●●○○ | ✅ | `pip install pyrnnoise` |
| `deepfilternet` | DeepFilterNet 3 | pretrained | fast | ●●●●● | ✅ | `pip install deepfilternet` |
| `demucs_denoiser` | Demucs Denoiser (DNS64) | pretrained | medium | ●●●●○ | ✅ | `pip install denoiser` |
| `demucs_vocals` | Demucs vocal isolation | pretrained | slow | ●●●●● | ✅ | `pip install demucs` |
| `resemble_enhance` | Resemble Enhance | pretrained | slow | ●●●●● | ✅ | `pip install resemble-enhance` |
| `local_unet` | **Your** SpectralUNet | local‑trained | fast | ●●●●○ | ✅ | train it (§5) |
| `api_huggingface` | Hugging Face API | api | medium | ●●●●○ | ❌ **uploads audio** | free HF token |

Everything except `api_huggingface` runs entirely on your machine.
`api_huggingface` needs `HF_TOKEN` plus a deployed compatible denoising endpoint,
configured via `HF_DENOISE_URL` or a hosted `HF_DENOISE_MODEL`. The UI
shows an "uploads audio" badge next to it.

### Which one should I pick?

```
Is the background MUSIC or a TV?           → demucs_vocals
Is it hiss / fan / hum / traffic?          → deepfilternet   (standalone binary needs no torch)
Is it clicks, slams, keyboard?             → demucs_denoiser
Is the recording clipped / phone-codec'd?  → resemble_enhance
Is speech preservation essential?         → keep the original; compare conservative DSP against it
Is it 3 hours long on an old laptop?       → rnnoise
Do you have your own domain data?          → local_unet
No GPU, no installs, need it now?          → spectral_gate
```

### Details worth knowing

**`spectral_gate`** — tracks a local minimum-statistics noise-power profile,
uses continuous soft power subtraction, and smooths the gain over time.
The default 64 ms window / 16 ms hop scales with sample rate. This avoids
the previous global binary-mask behavior and follows changing noise levels.
It will *not* separate another person talking. The old `noise_percentile`
argument is retained only for call compatibility and no longer controls the
adaptive profile.

**`wiener_mmse`** — the Ephraim–Malah log-spectral-amplitude estimator with
decision-directed *a priori* SNR and a speech-presence-gated noise tracker.
Suppression is limited (gain floored at −18 dB) which is deliberate: it sounds
natural rather than gating to silence. It can still distort speech, so
compare against the original; it is not a guarantee of forensic preservation.

> **Two bugs worth knowing about, both found by `scripts/benchmark.py`.**
> The textbook implementation scored **−9 dB SI-SDR** here — *worse than doing
> nothing* — for two reasons:
>
> 1. **The noise floor was seeded from the first 8 frames.** Textbooks assume a
>    recording opens with silence. Most real ones open mid-sentence, which seeds
>    the noise estimate with *speech* power, so the estimator then suppresses the
>    voice. Now initialised from local minimum-statistics noise estimates.
> 2. **Speech-presence probability was averaged across the spectrum.** Even
>    during a vowel most bins hold no speech, so the frame average read as
>    "silence", the noise tracker absorbed the voice, and suppression ran away.
>    The probability is now applied *per frequency bin* (the standard MCRA fix).
>
> With those fixed, and `n_fft`/`alpha_dd` chosen by sweeping against ground
> truth rather than by tradition (1024/0.92 rather than 512/0.98), it scores
> **+4.4 to +4.8 dB** across unseen SNRs and speaker counts — the best denoiser
> in the bare-install benchmark.

Those figures describe the earlier synthetic benchmark. The current defaults
use sample-rate-scaled windows and adaptive profiles; see AUDIO_VALIDATION.md
for the newer recorded-speech validation (+7.58 dB mean across 54 unseen DSP
cases). Neither benchmark guarantees performance on arbitrary recordings.

**`deepfilternet`** — two-stage: an ERB gain envelope over the whole band, then
complex *deep filters* over the first ~5 kHz which can actually recover phase.
48 kHz, ~2.3 M params, ~0.05× real time on one CPU core. This is the default
recommendation.

**`demucs_vocals`** — not a denoiser at all: htdemucs splits the audio into
drums/bass/other/vocals and we keep the vocals. When the interference is music,
this beats every actual denoiser by a wide margin. The result JSON includes
`stem_energy_share`, so you can see how much of the file was music.

**`local_unet`** — see §5.

---

## 3. Reading the metrics

Every run reports these. **All of them are reference-free estimates** — we do
not have the clean ground truth for an arbitrary upload, so nothing here is
PESQ-grade. They are good for *comparing runs on the same file*.

| metric | meaning | good value |
|---|---|---|
| `snr_before_db` / `snr_after_db` | VAD-based SNR: speech-frame power vs silence-frame power | — |
| `snr_improvement_db` | the difference | > 5 dB is a real improvement |
| `noise_reduction_db` | how far the noise floor dropped | 10–25 dB typical |
| `speech_preserved` | correlation of the 300–3400 Hz energy envelope, before vs after, over speech frames | **> 0.9** = words intact; < 0.7 = the denoiser is eating speech |
| `speech_band_energy_ratio` | raw energy ratio in that band | informational only |
| `quality_score` | 45 % SNR gain + 30 % noise drop + 25 % speech preserved | 0–100 |
| `clipping_ratio` | fraction of samples at full scale | should be ~0 |

### Why `speech_preserved` is a correlation, not a ratio

The obvious metric — "how much speech energy survived" — punishes good
denoisers: a denoiser is *supposed* to remove energy, so the ratio drops as soon
as the hiss goes. What actually signals damage is the speech **envelope**
changing shape. If the frame-by-frame energy contour of the telephone band still
tracks the original during speech, the words are intact no matter how much noise
was stripped. That is what we measure.

### If you *do* have a clean reference

`src/core/metrics.py` also provides real metrics for evaluation work:

```python
from src.core.metrics import si_sdr, pesq_score, stoi_score
si_sdr(estimate, reference)          # dB, always available
pesq_score(reference, estimate)      # needs: pip install pesq
stoi_score(reference, estimate)      # needs: pip install pystoi
```

---

## 4. Adding your own backend

Three steps, no registration file to edit:

```python
# src/denoising/methods/my_method.py
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ..base import BaseDenoiser

@register_denoiser
class MyDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="my_method", name="My Method", kind="denoise", family="dsp",
        description="…", pip=["some_package"], install_hint="pip install some_package",
    )
    target_sr = 16000          # base class resamples in and back out for you

    def _denoise(self, audio: AudioBuffer):
        cleaned = ...          # numpy float32, same rate as audio.sr
        return AudioBuffer(cleaned, audio.sr), {"backend": "mine"}
```

Then add `"my_method"` to `MODULES` in `methods/__init__.py`. The base class
handles timing, resampling, NaN guarding, length matching, metrics and error
capture; the registry handles availability probing so a missing dependency shows
up as a greyed-out card in the UI instead of a crash.

---

## 5. Training the local model

> **Nothing trains automatically.** The app never starts a training run. You
> start it, explicitly, with the command below.

### 5.1 The architecture (`src/denoising/architectures.py`)

`SpectralUNet` — a masking U-Net over the log-magnitude STFT with a bottleneck
BiGRU for temporal context.

```
wav ─► STFT ─► log1p|X| ─► stem ─► down×3 ─► [BiGRU] ─► up×3 (+skips) ─► 1×1 conv ─► mask
                                                                                     │
wav ◄────────────────────────── iSTFT ◄─── mask · X (complex) ◄──────────────────────┘
```

| preset | params | sample rate | crop | notes |
|---|---|---|---|---|
| `tiny` | ~0.35 M | 8 kHz | 1.0 s | the only one that is sane on a CPU |
| `base` | ~1.6 M | 16 kHz | 2.0 s | project default; what `local_unet` expects |
| `large` | ~4.5 M | 16 kHz | 4.0 s | needs a GPU |

Loss = **SI-SNR** (waveform, scale invariant) + 0.5 × **multi-resolution STFT**
(magnitude at 512/1024/2048). The STFT term is what prevents the "underwater"
artefacts a pure time-domain loss allows.

### 5.2 Data

Mixtures are generated **on the fly** — you never store a noisy corpus, and
every epoch sees new combinations (worth several dB versus a fixed set).

```
data/datasets/clean/**/*.wav     clean speech      (required)
data/datasets/noise/**/*.wav     noise recordings  (optional)
```

If `noise_dir` is empty the loader synthesises white / pink / brown / mains-hum /
impulsive noise, so **you can start training with speech alone**.

Free corpora (see `data/README.md` for links and sizes):

| purpose | corpus | size | note |
|---|---|---|---|
| clean | LibriSpeech `train-clean-100` | 6 GB | the standard starting point |
| clean | VoiceBank (VCTK) | 11 GB | pairs with DEMAND for the classic benchmark |
| clean | Common Voice | varies | best for non-English |
| noise | DEMAND | 4 GB | 18 real environments |
| noise | MUSAN | 11 GB | noise + music + babble |
| noise | WHAM! | 76 GB | recorded in real cafés/bars |
| noise | ESC-50 | 600 MB | small, fine for a first run |

A first useful model needs surprisingly little: **~10 hours of clean speech and
~2 hours of noise** already gives a model that beats `spectral_gate` on its own
domain.

### 5.3 Commands

```bash
pip install -r requirements-train.txt

# 1) ALWAYS do this first -- it times real steps on YOUR machine and prints the
#    true ETA, then exits without training anything.
python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml --estimate-only

# 2) start for real
python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml

# CPU-friendly variant
python -m src.denoising.training.train_unet --preset tiny --clean-dir data/datasets/clean

# stop cleanly after 6 hours (the checkpoint is still written)
python -m src.denoising.training.train_unet --config … --max-hours 6

# resume
python -m src.denoising.training.train_unet --config … --resume models/checkpoints/denoise_unet_last.pt
```

Outputs: `models/checkpoints/denoise_unet_{last,best}.pt` and TensorBoard logs
in `runs/denoise_unet/`. As soon as `denoise_unet_best.pt` exists, the method
**Local SpectralUNet (your model)** turns green in the UI.

### 5.4 How long does training take?

Estimates for a full run, assuming the dataset is on a local SSD. **The script
prints a measured number for your actual machine before it starts — trust that
one over this table.**

| preset | steps/epoch × epochs | CPU (4 cores, e.g. i7 laptop) | RTX 3060 / T4 | A100 / 4090 |
|---|---|---|---|---|
| `tiny` (0.35 M, 8 kHz, 1 s, bs 16) | 400 × 30 = 12 k | ≈ 1.2 s/step → **~4 hours** | ≈ 0.06 s/step → **~15 min** | **~5 min** |
| `base` (1.6 M, 16 kHz, 2 s, bs 8) | 1 000 × 100 = 100 k | ≈ 6 s/step → **~7 days** ⚠ not practical | ≈ 0.25 s/step → **~7 hours** | **~2 hours** |
| `large` (4.5 M, 16 kHz, 4 s, bs 8) | 2 000 × 200 = 400 k | ⚠ not practical | ≈ 0.9 s/step → **~4 days** | **~1 day** |

Per-epoch, so you can plan a partial run:

| preset | CPU / epoch | RTX 3060 / epoch | A100 / epoch |
|---|---|---|---|
| `tiny` | ~8 min | ~25 s | ~10 s |
| `base` | ~1.7 h | ~4 min | ~1.2 min |
| `large` | ~7 h | ~30 min | ~7 min |

**Good news for impatience:** denoisers converge fast. On the `base` preset you
usually get 80 % of the final quality within the **first 10–15 epochs**
(~1 hour on a 3060). Use `--max-hours` and keep `denoise_unet_best.pt`.

### 5.5 What quality to expect

Typical numbers on standard test sets (literature values for comparable
architectures — your mileage depends entirely on your data):

| model | SI-SDR improvement | PESQ (VoiceBank-DEMAND) |
|---|---|---|
| `spectral_gate` | +3 to +6 dB | ~2.1 |
| `wiener_mmse` | +4 to +7 dB | ~2.2 |
| RNNoise | +6 to +9 dB | ~2.3 |
| **SpectralUNet `tiny`, 4 h of training** | +6 to +8 dB | ~2.3 |
| **SpectralUNet `base`, converged** | +9 to +12 dB | ~2.6–2.9 |
| DeepFilterNet 3 | +13 to +16 dB | ~3.1 |

Beating DeepFilterNet in general is not realistic — it was trained on ~500 h
with far more compute. Beating it **on your own recording conditions** is very
realistic, and that is the point of the local model.

### 5.6 Tuning notes

* **SNR range** (`snr_range: [-5, 20]`) — widen the low end if your audio is
  really bad; a model only learns to fix SNRs it has seen.
* **`mask_type: crm`** — complex ratio mask; ~0.5 dB better than `sigmoid`,
  slightly slower, occasionally unstable early in training.
* **`use_gru: false`** — ~25 % faster, ~1 dB worse. Worth it on CPU.
* **`w_stft: 0`** — pure SI-SNR. Trains faster, sounds worse. Keep it at 0.5.
* **Loss stops improving** — halve the LR (the scheduler does this after 5 flat
  epochs) or check that your noise corpus is actually varied.
* **Output sounds "phasey"** — increase `w_stft`, or switch to `crm`.
* **Windows + `num_workers > 0`** — must be guarded by `if __name__ ==
  "__main__"`, which the script already does; if you still get spawn errors,
  set `num_workers: 0`.

---

## 6. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| every neural method is greyed out | no torch | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| `pip install torch` fails | Python 3.13/3.14 — wheels lag | make a Python 3.11 venv for the neural stack; the DSP methods work on any version |
| DeepFilterNet first run is slow | downloading ~2 MB of weights | one-off |
| output is quieter than the input | correct — the noise is gone | check `speech_preserved`, not loudness |
| output sounds robotic | over-suppression | use `wiener_mmse`, or lower `prop_decrease` in `spectral_gate` |
| `local_unet` says "no trained checkpoint" | you have not trained it | §5, or pick another method |
| Hugging Face method 503s | free-tier cold start | it retries; the model needs ~20 s to load |

---

## 7. References

- Ephraim & Malah (1985), *Speech enhancement using a minimum mean-square error log-spectral amplitude estimator*, IEEE TASSP.
- Valin (2018), *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement* (RNNoise), MMSP.
- Défossez et al. (2020), *Real Time Speech Enhancement in the Waveform Domain*, INTERSPEECH.
- Schröter et al. (2022/2023), *DeepFilterNet2 / DeepFilterNet3*, INTERSPEECH / ICASSP.
- Yamamoto et al. (2020), *Parallel WaveGAN* — the multi-resolution STFT loss used here.
- Le Roux et al. (2019), *SDR – half-baked or well done?*, ICASSP — the SI-SDR definition.
- Rouard et al. (2023), *Hybrid Transformers for Music Source Separation*, ICASSP (htdemucs).

**Next stage:** [`../separation/README.md`](../separation/README.md)
