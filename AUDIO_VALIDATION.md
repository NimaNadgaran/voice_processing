# Audio tuning and validation — 2026-09-30

Checked the denoisers, separators, shared audio handling, complete local
pipelines and sequential training launcher. "Runs successfully" means the output
is finite and has the original sample rate and length; it does not mean perfect
speech preservation or speaker isolation. The numerical results below use actual
recorded LibriSpeech voices with known references, not reference-free UI scores.
They are small reproducible checks, not a general evaluation across microphones,
languages, music, reverberation or long conversations. No subjective listening
review has been performed.

## Denoising

Adaptive local noise tracking replaced the gate's global binary thresholds.
Both DSP paths use sample-rate-independent 64 ms analysis windows. MMSE-LSA's
noise correction was swept to reduce high-SNR speech damage. Short clips and
edge smoothing preserve the timeline. DNS64 and local U-Net now use contextual
crossfaded chunks. RNNoise actually flushes its delayed frames before trimming
latency and supports the installed wrapper without its incompatible resampler.
Demucs handles silence and unbiased normalization; Resemble enhancement no
longer denoises twice and explicitly reports non-generative fallback.

The unseen DSP check covered 54 combinations: three held-out speakers,
white/hum/changing noise, 5/15 dB input SNR and 8/44.1/48 kHz audio (seed 99).

| Denoiser | Mean SI-SDR improvement | Worst case |
|---|---:|---:|
| Adaptive spectral gate | +4.99 dB | +1.53 dB |
| MMSE-LSA | +7.58 dB | +1.51 dB |

An independent six-case neural check (seed 41, single recorded speaker,
16/48 kHz and three noise types at 5 dB) measured RNNoise +7.03 dB mean,
DNS64 +13.35 dB, and the existing local U-Net +10.28 dB. In the all-backend
single-speaker audit, DeepFilterNet also improved the reference (+8.97 dB).

Important overlap limitation: on the two-speaker added-noise stress case,
DeepFilterNet (-1.25 dB), DNS64 (-3.79 dB) and RNNoise (-0.63 dB) reduced
reference quality. These speech-enhancement models can treat a second voice
as interference. Do not assume a denoiser suitable for one voice is suitable
before overlapping-speech separation. Clipping, music and transients still
need representative data for quality claims.

Exported a 12-second clip at 15 seconds from the bundled NYC vlog through
all seven installed non-control denoisers. Outputs decode and preserve the
timeline; the recording has no clean reference, so no true gain is reported.
Files: `data/outputs/denoise_tuning_vlog_verified/recording_*.wav`.

## Separation and complete pipelines

Fixed the Asteroid LibriMix models' sample-rate mismatch: their serialization
can report an 8 kHz constructor default, but the published checkpoints require
16 kHz. Known model rates now override that misleading default. SepFormer
uses checkpoint sample rate/source count; neural adapters pad tiny inputs.
Chunk stitching handles padded final blocks, original lengths and changing
source counts. Source cleanup retains speaker labels and short quiet
diarized turns instead of energy-sorting their identity away. Diarization
overlap is measured from actual intersecting turns, not total talk time.
ClearVoice handles source/channel layouts; NeMo honors speaker hints and
supported diarization signatures; pyannote selects a version-compatible
pipeline and supports the 4.x result wrapper.

Two recorded speakers, seed 99, approximately 4 seconds of speech each:

| Separator | Overlap SI-SDRi | Turn-taking SI-SDRi |
|---|---:|---:|
| Asteroid 16 kHz Conv-TasNet | +16.41 dB | +41.08 dB |
| Existing local Conv-TasNet | +9.18 dB | +13.36 dB |
| SepFormer WHAMR 16 kHz | +4.40 dB | +4.39 dB |
| Clustering diarization | -2.44 dB | +29.63 dB |

Also ran the actual three-source checkpoints: Asteroid improved overlap by
+6.36 dB and turn-taking by +14.22 dB; SepFormer WSJ03Mix improved them by
+8.40 dB and +0.80 dB respectively. This checks the three-source selection
paths too, and shows why no one checkpoint should be called universally best.

Clustering is a turn-attribution method: it cannot acoustically unmix voices
speaking simultaneously. Fixed-source neural models are limited to their
trained source count; speaker identity can still be ambiguous when successive
chunk overlaps contain silence. Restoring the input sample rate cannot restore
bandwidth removed by an 8 kHz local checkpoint. For higher bandwidth, train a
16 kHz denoiser via the new compact/default CPU setup.

Complete preset pipelines were tested with exported WAVs and known references
on noisy overlap (5 dB) and clean turn-taking. All seven ungated local presets
completed. The previous Cocktail party preset (DeepFilterNet → SepFormer)
scored +1.77 dB on noisy overlap versus +11.40 dB for MMSE-LSA → Asteroid.
The preset now uses the latter, with SepFormer as its dependency fallback.
This is a tested default for overlap, not a universal winner. Clean turn-taking
usually benefits less from denoising; the no-denoising control stays available.

## Access and dependency limits

| Backend | Verification status on this computer |
|---|---|
| DSP, RNNoise, DeepFilterNet binary, DNS64, htdemucs, local U-Net | Real inference and format checks |
| Clustering, Asteroid, SepFormer, local Conv-TasNet | Real inference and reference-based two-speaker checks |
| Resemble Enhance | Adapter regression tests; package not installed |
| ClearVoice/MossFormer | Adapter shape/configuration tests; package not installed |
| NeMo MSDD | Manifest/config/RTTM adapter tests; package not installed |
| pyannote | Adapter tests; account has not accepted/accessed community-1 weights |
| Hugging Face methods | Request/response/error tests; no real recording uploaded |

For pyannote 4.x, accept the conditions for
`pyannote/speaker-diarization-community-1` and set a token for that account.
For 3.x use `speaker-diarization-3.1` and `segmentation-3.0`. An installed
package and token alone do not establish access to gated weights.

Cloud inference uses the current router, independent in-memory WAV payloads,
bounded warmup/rate-limit retries, informative access errors, and correct
resampling when returned sources have different rates. Set `HF_TOKEN` plus
`HF_DENOISE_URL` / `HF_SEPARATION_URL` to deployed compatible endpoints, or
explicitly select models that are actually hosted. Hub availability is not
hosted inference availability; costs may apply. No cloud quality claim is made.

## Local training and ETA verification

`python setup.py` prepares an environment and training data, then times and
trains `local_unet` followed by `local_convtasnet`. `train_models(models=None)`
is the callable entry point; None means all local trainable models.

```bash
python setup.py --estimate-only
python setup.py --models local_unet,local_convtasnet --resume
python setup.py --clean-dir /path/to/speech --speakers-dir /path/to/speakers
```

Without existing training speech, setup downloads LibriSpeech train-clean-100
(~6.4 GB; allow 15 GB free disk) and reuses it for both models. Explicit missing
data paths fail rather than silently downloading into a different location.
Default automatic reuse excludes dev/test corpora. CPU chooses compact 16 kHz
U-Net and tiny Conv-TasNet; CUDA chooses base presets. A fresh clone still
needs internet access and a Python interpreter supported by the dependencies.
This is a training launcher, not a setuptools package-installer script.

Timing probes include data loading and validation, restore model weights,
buffers, optimizer/scaler and torch RNG, and do not save probe weights as
trained checkpoints. Queue ETAs include measured subprocess/probe startup;
per-epoch updates reflect actual training and validation time. Dependency
installation and data preparation precede the training estimate. Download
progress reports speed and remaining time. ETAs remain estimates and change
with hardware load, data speed and validation cost.

Completed an isolated end-to-end setup run for both models: one epoch, two
training steps and four validation items each. This verifies sequential
execution, real optimization, checkpoints and ETA reports; it is not full
quality training. The original app checkpoints were not overwritten.
Artifacts: `data/outputs/setup_verification/checkpoints/` and
`runs/setup_20260930-194250_c11f59/plan.json`.

Also verified a time-budget stop after one step in each model: both checkpoints
recorded epoch 0, rather than incorrectly skipping the unfinished first epoch.
Resumed that queue and completed both models sequentially; their final
checkpoints recorded epoch 1 with two training steps in the resumed epoch.
Resume verification plan: `runs/setup_20260930-202214_7c0d52/plan.json`.

Fresh training backs up existing best/last checkpoints in the run folder.
`--resume` restores checkpoint configuration, optimizer, scheduler, scaler
and prior best validation score. Time-budget stops save partial weights;
resume repeats the unfinished epoch. Ctrl+C resumes from the latest saved
checkpoint, not necessarily the interrupted step. `--estimate-only` installs
dependencies/prepares data unless `--no-install --no-download` is supplied.

## Reproducing checks

```bash
python -m pytest tests -q
python scripts/benchmark_denoising.py --methods spectral_gate,wiener_mmse --clips 3 --seconds 4 --rates 8000,44100,48000 --snrs 5,15 --seed 99 --label final_dsp_unseen
python scripts/audit_audio.py --save-audio --label verified
python scripts/audit_audio.py --methods sepformer,convtasnet_asteroid --speakers 3 --save-audio --label three_speakers
python scripts/benchmark_paths.py --paths path1,path2,path3,path4,path5,path6,path7,path8,path9 --label verified
```

The audio-stage pytest suite passed **129 tests** before speech-to-text was
added. The later combined suite passed **168 tests**, documented in
[TRANSCRIPTION_VALIDATION.md](TRANSCRIPTION_VALIDATION.md).
Audio tests cover short/empty audio,
sample rates, delayed-tail preservation, chunk permutation and lengths,
backend result layouts, label/turn identity, cloud isolation and access errors,
training-state preservation, model selection and validation-inclusive ETAs.
Optional adapter tests require the corresponding numerical test dependencies.
Reports and WAVs are under `data/outputs/`; do not commit datasets, pretrained
weights or generated audio merely to publish the code changes.
