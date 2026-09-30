"""Long-form explanations for every denoiser, shown in the UI's "How this works".

Why this lives in one file rather than inside each ``MethodInfo``: it is
*documentation*, it is long, and keeping it together is what makes the tone and
the level of detail consistent across ten methods. The code for each backend
still lives on its own in ``methods/``; this only supplies prose.

Every entry answers the same four questions:

* ``how_it_works`` -- what the method actually does to your audio
* ``steps``        -- the algorithm, in order, no hand-waving
* ``strengths``    -- when to reach for it
* ``limitations``  -- stated honestly, because picking the wrong tool is the
                      single most common reason people are disappointed
"""

from __future__ import annotations

from typing import Any, Dict

DETAILS: Dict[str, Dict[str, Any]] = {

    # ---------------------------------------------------------------- none --
    "none": {
        "how_it_works": (
            "It does nothing at all -- the audio passes through untouched. That is "
            "the point: it is the control condition. Run it as a second path next "
            "to a real denoiser and the comparison table tells you whether the "
            "denoiser actually helped the speaker separation, or quietly ate the "
            "quietest speaker."
        ),
        "steps": [
            "Copy the input to the output.",
            "Still measure everything, so the numbers line up with the other paths.",
        ],
        "strengths": ["Free", "The honest baseline for any comparison"],
        "limitations": ["Removes no noise whatsoever"],
        "latency": "instant",
        "reference": "",
    },

    # ------------------------------------------------------- spectral_gate --
    "spectral_gate": {
        "how_it_works": (
            "The oldest trick in audio restoration, done carefully. It listens to "
            "the quietest moments of your recording to learn what the noise sounds "
            "like at every frequency -- a 'noise print' -- then goes through the "
            "whole file and turns down any time-frequency spot that is not clearly "
            "louder than that print. Because hiss, fans and hum sit at a roughly "
            "constant level while speech comes and goes, the gate closes on the "
            "noise and opens for the voice."
        ),
        "steps": [
            "Cut the audio into overlapping 64 ms windows and take an FFT of each "
            "(a spectrogram: frequency on one axis, time on the other).",
            "For every frequency bin, take the 12th-percentile loudness over the "
            "whole file. That is the noise floor at that frequency.",
            "Set a threshold 1.5 standard deviations above the floor.",
            "Build a mask: 1 where the signal is above the threshold, 0 below.",
            "Blur that mask over 4 frequency bins and 6 time frames. This is the "
            "step that matters -- an unsmoothed mask produces 'musical noise', the "
            "random bubbling chirps that make naive noise removal sound terrible.",
            "Multiply the spectrogram by the mask and invert the FFT back to audio.",
        ],
        "strengths": [
            "Instant, and needs no download or GPU",
            "Very effective on steady noise: fans, air-conditioning, hiss, mains hum",
            "Never invents sound that was not there",
        ],
        "limitations": [
            "Assumes the noise stays roughly constant -- it cannot follow a passing motorbike",
            "Will not remove another person talking (that is not noise, it is speech)",
            "Push it too hard and consonants start to sound gated and lispy",
        ],
        "latency": "about 0.01x real time (a 10 minute file in ~6 seconds)",
        "reference": "Classic spectral subtraction; the optional `noisereduce` package implements the same idea",
    },

    # --------------------------------------------------------- wiener_mmse --
    "wiener_mmse": {
        "how_it_works": (
            "A statistical estimator rather than a gate. For every frequency, at "
            "every instant, it asks: 'given how loud this bin is now, and what I "
            "believe the noise level to be, what is the most likely loudness of the "
            "clean speech underneath?' -- and applies exactly that much gain. The "
            "1985 Ephraim-Malah result is that doing this in the *log* amplitude "
            "domain matches how we hear, which is why it still sounds better than "
            "many newer methods despite being pure maths with no training data."
        ),
        "steps": [
            "STFT the signal into 32 ms frames.",
            "Track the noise power per frequency with a recursive average that is "
            "frozen while speech is present, so it keeps adapting to changing noise "
            "without swallowing the voice.",
            "Estimate the 'a priori SNR' with the decision-directed approach: "
            "blend 98% of the previous frame's clean estimate with 2% of the "
            "current measurement. This heavy smoothing is precisely what removes "
            "musical noise.",
            "Apply the MMSE log-spectral-amplitude gain, G = xi/(1+xi) * exp(E1(v)/2), "
            "floored at -18 dB so residual noise stays natural instead of turning "
            "into unnatural silence.",
            "Inverse STFT with overlap-add.",
        ],
        "strengths": [
            "The most natural sounding option here -- no artefacts, ever",
            "Adapts to noise that changes slowly over time",
            "Deterministic maths: nothing is invented, which matters for evidential audio",
            "Real time, no model, no download",
        ],
        "limitations": [
            "Gentle by design: expect 6-12 dB, not the 20 dB a neural model gives",
            "Cannot touch babble, music or other speakers",
        ],
        "latency": "about 0.05x real time",
        "reference": "Ephraim & Malah, 'Speech enhancement using a MMSE log-spectral amplitude estimator', IEEE TASSP 1985",
    },

    # ------------------------------------------------------------- rnnoise --
    "rnnoise": {
        "how_it_works": (
            "A deliberately tiny neural network -- 85 kilobytes, about 88,000 "
            "weights -- that powers noise suppression in WebRTC, OBS and Mumble. "
            "The trick is that it does not try to learn audio from scratch. "
            "Classical DSP does the heavy lifting (band splitting, pitch analysis), "
            "and the network only decides *how much to turn down each of 22 "
            "frequency bands*, every 10 milliseconds. Predicting 22 numbers instead "
            "of a full spectrum is why something this small works at all."
        ),
        "steps": [
            "Split each 10 ms frame into 22 bands spaced like human hearing (Bark scale).",
            "Compute cepstral coefficients, their time derivatives, and pitch features.",
            "A 3-layer gated recurrent network (GRU) predicts one gain per band, "
            "plus a voice-activity probability which we surface as a VAD curve.",
            "Interpolate the 22 band gains across all frequency bins and apply them.",
            "A pitch comb filter cleans up the noise sitting between voice harmonics.",
        ],
        "strengths": [
            "Almost free: well under 1% of one CPU core in real time",
            "No torch, no GPU, no model download -- it ships inside the wheel",
            "Ideal for very long recordings on modest hardware",
            "Gives you a free per-frame speech probability",
        ],
        "limitations": [
            "Clearly weaker than DeepFilterNet on hard noise",
            "Trained mostly on 48 kHz speech + common noise; unusual material fares worse",
            "22 bands is coarse -- it cannot carve out narrow interference",
        ],
        "latency": "about 0.005x real time (an hour of audio in ~20 seconds)",
        "reference": "Valin, 'A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement', MMSP 2018",
    },

    # ------------------------------------------------------- deepfilternet --
    "deepfilternet": {
        "how_it_works": (
            "The best all-round denoiser in this project, and worth understanding. "
            "Most neural denoisers predict one gain per time-frequency bin, which "
            "throws away phase and blurs anything narrower than a bin. "
            "DeepFilterNet does two different things instead. For the whole "
            "spectrum it predicts gains on 32 perceptual ERB bands -- coarse, cheap, "
            "and enough for the high frequencies where we cannot hear fine detail. "
            "Then, for the low band (roughly up to 5 kHz, where the voice lives), it "
            "predicts a small complex *filter* that mixes the current frame with "
            "several neighbouring frames. That 'deep filtering' step can reconstruct "
            "phase and recover periodic structure a per-bin gain physically cannot -- "
            "which is why speech comes out crisp instead of muffled."
        ),
        "steps": [
            "Resample to 48 kHz and take a 20 ms STFT.",
            "Encoder: convolutional layers over the ERB features plus the complex "
            "low-band spectrum, with grouped GRUs for temporal context.",
            "Branch 1 predicts 32 real ERB gains, applied across the full band.",
            "Branch 2 predicts complex deep-filter coefficients for the low band; "
            "each output bin becomes a weighted sum of a few neighbouring frames.",
            "Combine both, inverse STFT, resample back to your file's rate.",
        ],
        "strengths": [
            "State of the art quality for its size -- around 2.3 M parameters",
            "Full 48 kHz band, so it does not dull the recording",
            "Removes fans, keyboards, traffic, and shortens reverb tails",
            "Fast enough for real time on one CPU core",
        ],
        "limitations": [
            "The pip package needs torch (~2.5 GB) and Python <=3.11; the "
            "standalone binary needs neither",
            "Trained on the DNS corpus: mostly English speech and common noise types",
            "It is a suppressor, not a separator -- another talker stays put",
        ],
        "latency": "about 0.05x real time on CPU",
        "reference": "Schroeter et al., 'DeepFilterNet2 / DeepFilterNet3', INTERSPEECH 2022 / ICASSP 2023",
    },

    # ----------------------------------------------------- demucs_denoiser --
    "demucs_denoiser": {
        "how_it_works": (
            "Meta's waveform-domain denoiser. Unlike everything above it never "
            "converts to a spectrogram at all -- it is a U-Net that reads raw "
            "samples, squeezes them through an encoder, passes the bottleneck "
            "through an LSTM for context, and decodes clean samples back out. "
            "Working directly on the waveform means phase is never discarded, "
            "which is why it handles sharp transient noises (a door slam, a "
            "keyboard click, a cough) that spectral methods smear across a whole "
            "analysis window."
        ),
        "steps": [
            "Resample to 16 kHz.",
            "Encoder: 5 strided 1-D convolution blocks, each halving the time "
            "resolution and doubling the channels.",
            "A 2-layer LSTM at the bottleneck supplies long-range context.",
            "Decoder: 5 transposed-convolution blocks with skip connections back "
            "to the matching encoder layer.",
            "Long files are processed in 60 s blocks with 0.5 s overlap.",
        ],
        "strengths": [
            "Best in this project on sudden, non-stationary noise",
            "No spectral smearing, because there is no spectrogram",
            "Trained on the large DNS challenge dataset",
        ],
        "limitations": [
            "Works at 16 kHz, so anything above 8 kHz is not preserved -- it will "
            "dull a music-quality recording",
            "Heavier than DeepFilterNet for generally similar or worse results on steady noise",
            "Needs torch",
        ],
        "latency": "roughly 0.3x real time on CPU",
        "reference": "Defossez et al., 'Real Time Speech Enhancement in the Waveform Domain', INTERSPEECH 2020",
    },

    # ------------------------------------------------------- demucs_vocals --
    "demucs_vocals": {
        "how_it_works": (
            "Not a denoiser at all, used as one. htdemucs is a music source "
            "separation model: give it a song and it returns four stems -- drums, "
            "bass, other, and vocals. We throw away three and keep the vocals. "
            "This is by far the best way to rescue speech recorded over music or a "
            "TV, because to a denoiser music is not noise: it is structured, "
            "harmonic, non-stationary sound that every noise-print method reads as "
            "signal. A model trained specifically to tell voices from instruments "
            "succeeds where they all fail."
        ),
        "steps": [
            "Resample to 44.1 kHz and normalise.",
            "The hybrid transformer processes the signal in *two* domains at once: "
            "a spectrogram branch and a waveform branch.",
            "Cross-domain attention layers let the two branches share information.",
            "Four stems are decoded; we keep 'vocals' and report how much energy "
            "each stem held, so you can see how much of your file was music.",
            "Processing is windowed with 15% overlap and blended.",
        ],
        "strengths": [
            "The only reliable answer when the background is music or television",
            "Keeps full 44.1 kHz bandwidth",
            "The per-stem energy report tells you what was actually in the file",
        ],
        "limitations": [
            "Slow: roughly real time on CPU, so a 10 minute file takes ~10 minutes",
            "~80 MB download on first use",
            "Trained on *sung* vocals; on speech it occasionally clips quiet consonants",
            "Overkill (and slightly lossy) if the background is only hiss",
        ],
        "latency": "roughly 1x real time on CPU, ~0.1x on a GPU",
        "reference": "Rouard et al., 'Hybrid Transformers for Music Source Separation', ICASSP 2023",
    },

    # ---------------------------------------------------- resemble_enhance --
    "resemble_enhance": {
        "how_it_works": (
            "The only *generative* option here, and the distinction matters. Every "
            "other method can only remove or attenuate what is already in the file. "
            "This one runs a conventional denoiser first, then feeds the result to "
            "a conditional latent diffusion model that regenerates a clean 44.1 kHz "
            "waveform from scratch, conditioned on what it heard. Because it is "
            "synthesising rather than filtering, it can restore detail that is "
            "genuinely gone: clipped peaks, a missing high band from a phone codec, "
            "chunks lost to packet drops."
        ),
        "steps": [
            "Stage 1: a UNet denoiser removes the obvious noise.",
            "Stage 2: a CFM (conditional flow matching) diffusion model, guided by "
            "the denoised audio, generates clean latents over ~32 solver steps.",
            "A vocoder turns those latents back into a 44.1 kHz waveform.",
        ],
        "strengths": [
            "Repairs damage no mask-based method can: clipping, band-limiting, codec artefacts",
            "Output sounds studio-clean rather than merely 'less noisy'",
        ],
        "limitations": [
            "IT INVENTS DETAIL. The output is a plausible reconstruction, not evidence. "
            "Never use it for forensic, biometric or legal audio.",
            "Can subtly change a person's timbre",
            "Very slow on CPU (several times real time); a GPU is strongly advised",
            "~500 MB of weights",
        ],
        "latency": "several times real time on CPU; ~0.2x on a GPU",
        "reference": "resemble-ai/resemble-enhance (UNet denoiser + CFM enhancer)",
    },

    # ---------------------------------------------------------- local_unet --
    "local_unet": {
        "how_it_works": (
            "Your own model -- the one you train with the scripts in "
            "src/denoising/training/. It is a masking U-Net over the log-magnitude "
            "spectrogram with a bidirectional GRU at the bottleneck for temporal "
            "context, about 1.6 M parameters. The architecture is unremarkable on "
            "purpose; the value is the training data. A generic model has to work "
            "for everyone, so it is average everywhere. Train this one on twenty "
            "minutes of *your* room, *your* microphone and *your* noise, and on "
            "that material it can beat models a hundred times its size."
        ),
        "steps": [
            "STFT the noisy audio and take log(1+|X|).",
            "Encoder: 4 convolutional blocks, halving frequency resolution each time.",
            "A bidirectional GRU at the bottleneck adds temporal context.",
            "Decoder: 4 blocks with skip connections, ending in a 1x1 convolution.",
            "The output is a mask in [0,1] (or a complex ratio mask), multiplied "
            "into the original complex spectrogram.",
            "Inverse STFT. Long files are processed in 10 s blocks, cross-faded.",
        ],
        "strengths": [
            "Tunable to your exact recording conditions",
            "Completely offline; nothing is downloaded and nothing phones home",
            "Small and fast at inference",
            "A readable reference implementation you can modify",
        ],
        "limitations": [
            "You have to train it first -- it is unavailable until the checkpoint exists",
            "Only as good as the data you give it; it will not generalise beyond that",
            "Training the default preset is impractical on a CPU (see the README's time table)",
        ],
        "latency": "about 0.1x real time on CPU once trained",
        "reference": "This project: src/denoising/architectures.py + training/train_unet.py",
    },

    # ----------------------------------------------------- api_huggingface --
    "api_huggingface": {
        "how_it_works": (
            "Instead of running a model on your machine, this uploads the audio to "
            "Hugging Face's free Inference API, which runs a hosted speech "
            "enhancement model and sends the cleaned audio back. It exists for the "
            "case where the local options are out of reach -- a very weak machine, "
            "or an environment where you cannot install a 2.5 GB torch wheel."
        ),
        "steps": [
            "Encode the audio as a 16 kHz WAV.",
            "POST it to the model endpoint with your free read token.",
            "If the model is cold (HTTP 503) wait for the estimated load time and retry, up to 4 times.",
            "Decode the returned audio (raw bytes or a base64 blob) back into the pipeline.",
        ],
        "strengths": [
            "No local compute and no large installs",
            "Free tier is enough for occasional files",
        ],
        "limitations": [
            "YOUR AUDIO LEAVES YOUR COMPUTER. It is uploaded to a third party. "
            "This method stays disabled until you set HF_TOKEN yourself.",
            "Rate limited, and slow to start when the model is cold",
            "Needs an internet connection; unusable offline",
            "Long files may exceed the endpoint's limits",
        ],
        "latency": "seconds to a couple of minutes, mostly network and cold start",
        "reference": "Hugging Face Inference API, audio-to-audio pipeline",
    },
}
