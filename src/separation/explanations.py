"""Long-form explanations for every separator, shown in the UI.

Same contract as ``src/denoising/explanations.py``. The most important thing
these entries communicate is the **diarization vs source separation** split,
because choosing the wrong family is the number one reason results disappoint:

* *Diarization* answers "who spoke when" and cuts the timeline. Any number of
  speakers, cheap, but during genuine overlap both voices land in both files.
* *Source separation* answers "unmix these voices" and can pull apart people
  talking at the same time -- but the number of outputs is baked into the
  model's weights (2 or 3), so it cannot give you four files.
"""

from __future__ import annotations

from typing import Any, Dict

DETAILS: Dict[str, Dict[str, Any]] = {

    # ---------------------------------------------------------------- none --
    "none": {
        "how_it_works": (
            "It does not split anything -- whatever reaches this stage is written "
            "out as a single track. Use it to build a denoise-only path (clean the "
            "file, keep it whole), for recordings with one speaker, or as the "
            "control that shows what the separator actually changed."
        ),
        "steps": [
            "Copy the incoming audio to one output track.",
            "Still run VAD and the usual measurements, so the numbers line up "
            "with the other paths in the comparison table.",
        ],
        "strengths": ["Free", "Never invents a second speaker", "The honest baseline for any comparison"],
        "limitations": ["Separates nothing -- one file in, one file out"],
        "latency": "instant",
        "reference": "",
    },

    # ------------------------------------------------------ diarize_cluster --
    "diarize_cluster": {
        "how_it_works": (
            "Diarization built from scratch in numpy, and the reason this app "
            "always works on a bare install. The idea: a person's voice has a "
            "consistent fingerprint -- pitch, and the resonances their throat and "
            "mouth impose on it. So chop the speech into short windows, turn each "
            "window into a vector describing that fingerprint, and cluster the "
            "vectors. Windows that land in the same cluster were spoken by the same "
            "person. Turn each cluster back into a time mask, multiply, and you have "
            "one file per speaker, perfectly aligned with the original timeline."
        ),
        "steps": [
            "Voice-activity detection: keep only the frames that contain speech.",
            "Cut the speech into 1.5 s windows (0.75 s hop) and embed each one. If "
            "SpeechBrain is installed we use ECAPA-TDNN speaker embeddings, which "
            "are trained precisely to separate voices; otherwise a numpy fallback "
            "of 12 MFCC means, 7 MFCC standard deviations, pitch (weighted x3 -- it "
            "is the strongest single cue) and two spectral-shape numbers.",
            "Decide how many speakers there are: cluster for every candidate K and "
            "score each partition with the silhouette coefficient, cross-checked "
            "against the eigengap of the affinity matrix.",
            "Re-embed at a finer 0.6 s resolution and assign each window to the "
            "nearest of the K centroids.",
            "Median-filter the label sequence over 5 windows -- people do not swap "
            "every 0.6 seconds, so isolated flips are errors.",
            "Merge neighbouring windows of the same speaker into turns, drop turns "
            "under 0.35 s, and build a per-speaker sample mask with 25 ms cosine "
            "fades so there are no clicks at the boundaries.",
            "Multiply the audio by each mask and order the speakers by who spoke first.",
        ],
        "strengths": [
            "Handles ANY number of speakers and works out the count itself",
            "No downloads, no torch, no GPU -- pure numpy",
            "Output is timeline-aligned, so the files stack in any DAW",
            "Gives real turn boundaries, which drives the timeline in the UI",
            "Automatically upgrades to proper ECAPA embeddings if SpeechBrain is installed",
        ],
        "limitations": [
            "This is diarization, not unmixing: while two people genuinely talk "
            "over each other, both voices appear in both files",
            "The numpy fallback embedding struggles with similar voices (two men "
            "of similar pitch); install speechbrain and it improves markedly",
            "Needs a few seconds of each speaker -- someone who says one word may be missed",
        ],
        "latency": "about 0.2x real time",
        "reference": "Standard embed-and-cluster diarization; ECAPA-TDNN from Desplanques et al., INTERSPEECH 2020",
    },

    # -------------------------------------------------------------- pyannote --
    "pyannote": {
        "how_it_works": (
            "The reference open-source diarization pipeline, and the best default "
            "for real meetings. It improves on plain clustering in one crucial way: "
            "a neural segmentation model looks at short sliding windows and outputs, "
            "for every instant, which speakers are active -- including *more than "
            "one at a time*. So unlike naive diarization it explicitly detects "
            "overlapped speech rather than forcing every moment to belong to one "
            "person. Those local decisions are then stitched together globally by "
            "clustering speaker embeddings, which is what lets it discover the "
            "number of speakers on its own."
        ),
        "steps": [
            "Slide a 10 s window across the audio.",
            "In each window, an end-to-end segmentation network (a SincNet + LSTM "
            "model) predicts per-frame activity for up to 3 local speakers, "
            "overlap included.",
            "Extract a speaker embedding for each local speaker in each window.",
            "Agglomerative hierarchical clustering ties local speakers across "
            "windows into global identities; the clustering threshold determines "
            "the final speaker count.",
            "Resolve the local decisions into a global timeline of turns.",
            "We convert those turns into one faded mask per speaker and write the files.",
        ],
        "strengths": [
            "Best real-world accuracy available for free (roughly 10-20% DER on hard data)",
            "Finds the number of speakers by itself, and accepts a pinned count",
            "Explicitly detects overlapping speech instead of ignoring it",
            "Handles any number of speakers",
        ],
        "limitations": [
            "You must accept the model licence on Hugging Face and set HF_TOKEN "
            "(both free, one-time)",
            "Still diarization: overlapped regions are attributed to all active "
            "speakers, not acoustically unmixed",
            "Needs torch",
        ],
        "latency": "about 0.3x real time on CPU, much faster on a GPU",
        "reference": "Bredin et al., 'pyannote.audio 2.1 / 3.1 speaker diarization pipeline', INTERSPEECH 2023",
    },

    # ------------------------------------------------------------ nemo_msdd --
    "nemo_msdd": {
        "how_it_works": (
            "NVIDIA's Multi-scale Diarization Decoder. Its insight is about window "
            "length: a long window gives a reliable speaker embedding but poor "
            "timing, while a short window times things precisely but the embedding "
            "is noisy. Instead of picking one, MSDD extracts embeddings at several "
            "scales at once (1.5 s, 1.0 s, 0.5 s and shorter) and learns how much "
            "to trust each scale at each moment. The decoder then estimates, for "
            "every pair of speakers, whether each is speaking -- so overlap falls "
            "out naturally instead of being a special case."
        ),
        "steps": [
            "Run VAD, then extract TitaNet speaker embeddings at multiple window scales.",
            "Cluster the longest-scale embeddings to get initial speaker profiles.",
            "The MSDD network weights the scales per frame and outputs pairwise "
            "speaker-presence probabilities.",
            "Thresholding those probabilities yields the final turns, including "
            "regions where two speakers are simultaneously active.",
        ],
        "strengths": [
            "Very strong on telephone audio and far-field meeting rooms",
            "Multi-scale fusion handles both fast turn-taking and stable identity",
            "Any number of speakers",
        ],
        "limitations": [
            "The heaviest install in the project by far -- NeMo pulls in a large tree",
            "Marked experimental here: NeMo's diarizer API has changed between "
            "releases, so we probe two call styles and fall back to parsing its RTTM",
            "Slow",
        ],
        "latency": "roughly 0.5-1x real time on CPU",
        "reference": "Park et al., 'Multi-scale Speaker Diarization with Dynamic Scale Weighting', INTERSPEECH 2022",
    },

    # ------------------------------------------------------------- sepformer --
    "sepformer": {
        "how_it_works": (
            "Genuine source separation -- the cocktail-party solver. It does not cut "
            "the timeline; it takes a mixture where two people speak simultaneously "
            "and reconstructs each voice as a full, continuous signal. It works on a "
            "*learned* representation rather than an FFT: a 1-D convolution turns "
            "the waveform into a latent 'spectrogram-like' code, a transformer "
            "predicts one mask per speaker in that code, and a transposed "
            "convolution turns each masked code back into audio. The dual-path "
            "trick makes attention affordable: the sequence is folded into chunks, "
            "and attention runs *within* each chunk and then *across* chunks, "
            "instead of over the whole (enormous) sequence at once."
        ),
        "steps": [
            "Encoder: a 1-D convolution with a short kernel produces a 256-channel "
            "latent representation.",
            "Fold the latent sequence into overlapping chunks.",
            "Intra-chunk transformer: models detail inside each chunk.",
            "Inter-chunk transformer: models long-range structure across chunks.",
            "Predict one mask per speaker; multiply into the latent.",
            "Decoder: transposed convolution reconstructs each speaker's waveform.",
            "Long files are cut into 10 s blocks with permutation stitching, so "
            "speakers never swap files between blocks.",
        ],
        "strengths": [
            "Actually unmixes simultaneous speech -- nothing else here except "
            "MossFormer2 and Conv-TasNet can",
            "Excellent SI-SDR on benchmark mixtures (~22 dB on WSJ0-2mix)",
        ],
        "limitations": [
            "The output count is fixed by the checkpoint: 2 or 3 speakers, never 4",
            "Memory and time grow quickly with length, hence the block processing",
            "Trained on clean anechoic mixtures; on a real noisy recording use the "
            "whamr16k checkpoint and denoise first",
            "Slow on CPU",
        ],
        "latency": "roughly 1-3x real time on CPU, ~0.1x on a GPU",
        "reference": "Subakan et al., 'Attention is All You Need in Speech Separation', ICASSP 2021",
    },

    # --------------------------------------------------- convtasnet_asteroid --
    "convtasnet_asteroid": {
        "how_it_works": (
            "The model that made time-domain separation work, and still the best "
            "speed/quality trade-off. Same three-part shape as SepFormer -- encoder, "
            "mask estimator, decoder -- but the mask estimator is a stack of dilated "
            "convolutions instead of attention. Each layer's dilation doubles "
            "(1, 2, 4, 8, ...), so a modest stack sees seconds of context while "
            "staying linear in length rather than quadratic. That is why it runs "
            "several times faster than SepFormer for most of the quality."
        ),
        "steps": [
            "Encoder: 1-D convolution, 16-sample kernel, stride 8 -> a 256-channel latent.",
            "Global layer normalisation, then a 1x1 convolution down to a 128-channel bottleneck.",
            "A temporal convolutional network: R repeats of X depthwise-separable "
            "blocks, dilation doubling within each repeat.",
            "A 1x1 convolution produces one mask per speaker; multiply into the latent.",
            "Decoder: transposed convolution back to waveforms.",
            "Trained with permutation-invariant SI-SNR (see the local model below).",
        ],
        "strengths": [
            "3-5x faster than SepFormer with most of the quality",
            "Small (~5 M parameters) and linear in input length",
            "The Asteroid zoo has 2- and 3-speaker checkpoints at 8 and 16 kHz",
            "The same architecture is what you train locally, so it is a fair reference point",
        ],
        "limitations": [
            "Fixed 2 or 3 outputs, like all separation models",
            "Slightly below SepFormer/MossFormer2 on benchmarks",
            "The clean-trained checkpoints degrade on noisy input -- denoise first",
        ],
        "latency": "roughly 0.3-1x real time on CPU",
        "reference": "Luo & Mesgarani, 'Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking', IEEE/ACM TASLP 2019",
    },

    # ------------------------------------------------ mossformer_clearvoice --
    "mossformer_clearvoice": {
        "how_it_works": (
            "One of the highest-scoring open separation models. MossFormer2 keeps "
            "the dual-path idea but replaces the expensive full attention with a "
            "gated single-head attention over joint local/global units, and adds a "
            "recurrent FSMN (feedforward sequential memory network) module to model "
            "fine-grained temporal detail that attention alone tends to smooth over. "
            "The result beats SepFormer on WSJ0-2mix while being cheaper per step."
        ),
        "steps": [
            "Convolutional encoder to a latent representation.",
            "Gated single-head attention with joint local and global units.",
            "An FSMN-based recurrent module refines detail between attention blocks.",
            "Mask prediction per speaker, then convolutional decoding.",
        ],
        "strengths": [
            "Best-in-class separation quality on standard benchmarks (>22 dB SI-SDRi)",
            "Packaged as a one-liner by ClearerVoice-Studio",
        ],
        "limitations": [
            "2 speakers only, at 16 kHz",
            "~200 MB of weights and genuinely slow without a GPU",
            "Like all separators, benchmark-trained: real noisy audio needs denoising first",
        ],
        "latency": "several times real time on CPU; a GPU is strongly advised",
        "reference": "Zhao et al., 'MossFormer2', ICASSP 2024 (ClearerVoice-Studio, Alibaba)",
    },

    # -------------------------------------------------------- local_convtasnet --
    "local_convtasnet": {
        "how_it_works": (
            "Your own Conv-TasNet, trained by the scripts in "
            "src/separation/training/. The architecture is the one described above; "
            "what is worth understanding is the *loss*, because separation training "
            "has a problem denoising does not: the model has no way to know which "
            "output should be which person. 'Alice, Bob' and 'Bob, Alice' are both "
            "correct, so training against a fixed order sends contradictory "
            "gradients and the model collapses to emitting the mixture twice. "
            "Permutation-invariant training fixes it: score every possible "
            "assignment of outputs to references and back-propagate only the best "
            "one. Train it on your own speakers and language and it can beat generic "
            "checkpoints on that material."
        ),
        "steps": [
            "Mixtures are generated on the fly: pick N *different* speakers, take a "
            "non-silent crop from each, scale to random relative levels, sum.",
            "Forward pass through encoder / TCN / decoder.",
            "Compute SI-SNR for every output-to-reference pairing.",
            "Keep the best permutation's mean SI-SNR; that is the loss (uPIT).",
            "Optionally add a mixture-consistency term: the separated sources should "
            "add back up to the input, which reduces energy vanishing on real audio.",
        ],
        "strengths": [
            "Tuned to your speakers, your language and your recording chain",
            "Fully offline, and fast at inference",
            "A complete, readable training recipe you can modify",
        ],
        "limitations": [
            "You must train it first, and separation converges much more slowly "
            "than denoising (see the time table in the README)",
            "The number of speakers is fixed at training time",
            "Needs 50+ distinct speakers of training data or it memorises voices "
            "instead of learning to separate",
        ],
        "latency": "about 0.3x real time on CPU once trained",
        "reference": "This project: src/separation/architectures.py + training/train_convtasnet.py",
    },

    # -------------------------------------------------------- api_huggingface --
    "api_huggingface": {
        "how_it_works": (
            "Uploads the audio to Hugging Face's free Inference API, which runs a "
            "hosted SepFormer and returns one audio blob per separated speaker. "
            "Useful when you cannot install a multi-gigabyte torch stack locally."
        ),
        "steps": [
            "Encode the audio as WAV and POST it to the model endpoint.",
            "Retry on HTTP 503 while the model cold-starts.",
            "Decode each returned base64 blob into a speaker track.",
        ],
        "strengths": ["No local compute", "No large installs", "Free tier"],
        "limitations": [
            "YOUR AUDIO LEAVES YOUR COMPUTER -- disabled until you set HF_TOKEN yourself",
            "2-3 speakers only, and rate limited",
            "Needs an internet connection",
        ],
        "latency": "seconds to minutes, mostly network and cold start",
        "reference": "Hugging Face Inference API, audio-to-audio pipeline",
    },
}
