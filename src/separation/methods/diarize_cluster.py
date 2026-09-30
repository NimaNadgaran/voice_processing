"""Method 1 -- Clustering diarization (the always-available separator).

No torch, no downloads, no training: pure numpy + the shared DSP layer.  This is
the method that guarantees the app produces "N speakers -> N files" on a bare
install, and it is the right tool for the most common real recording: a meeting
or interview where people mostly take turns.

Pipeline
--------
1. VAD -> speech only.
2. Coarse 1.5 s windows -> embeddings -> estimate K (see ``speaker_count.py``)
   unless the user pinned the number.
3. Fine 0.6 s windows -> embeddings -> assign to the K cluster centroids.
4. Median-filter the label sequence in time (people do not swap every 0.6 s),
   merge neighbouring segments of the same speaker, drop < 0.4 s islands.
5. Build one cosine-faded sample mask per speaker and multiply the waveform.

What you get
------------
Speaker i's file contains their turns and silence elsewhere -- perfectly aligned
with the original timeline, so you can stack the files in any DAW.

Honest limitation
-----------------
This is *diarization*, not source separation: during genuinely overlapping
speech, both voices land in both files.  If your recording is people talking
over each other, use SepFormer / Conv-TasNet / MossFormer2 instead, or run this
first and the neural separator on the overlapped regions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ...core.dsp import segments_to_mask
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ..base import BaseSeparator
from ..speaker_count import (
    DEFAULT_MAX_K,
    ecapa_embeddings,
    estimate_speaker_count,
    kmeans,
    speech_windows,
    window_features,
)


def _median_filter_labels(labels: np.ndarray, size: int = 5) -> np.ndarray:
    if len(labels) < size or size < 3:
        return labels
    half = size // 2
    padded = np.pad(labels, (half, half), mode="edge")
    out = np.empty_like(labels)
    for i in range(len(labels)):
        window = padded[i: i + size]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _merge_segments(segs: List[Tuple[float, float]], gap: float = 0.35) -> List[List[float]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for start, end in segs[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


@register_separator
class ClusterDiarizationSeparator(BaseSeparator):
    info = MethodInfo(
        key="diarize_cluster",
        name="Clustering Diarization (built-in)",
        kind="separate",
        family="dsp",
        description=(
            "Embeds short windows of speech, clusters them into speakers and writes "
            "one timeline-aligned file per speaker. No downloads, any number of "
            "speakers, works on a bare install."
        ),
        speed="fast",
        quality=3,
        needs_gpu=False,
        offline=True,
        pip=[],
        install_hint="no install needed (uses SpeechBrain ECAPA embeddings automatically if present)",
        notes="Turn-taking conversations: excellent. Simultaneous speech: it cannot split overlaps.",
        max_speakers=None,
    )
    target_sr = 16000
    restore_sr = True
    min_speech_seconds = 0.4

    def check_available(self) -> Tuple[bool, str]:
        return True, ""

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        est = estimate_speaker_count(
            audio, max_k=max(2, num_speakers or DEFAULT_MAX_K), use_embeddings=True
        )
        k = int(num_speakers or est["n_speakers"])
        k = max(1, k)

        fine_windows, _ = speech_windows(audio, win_s=0.6, hop_s=0.3, min_speech_frac=0.5)
        if not fine_windows or k == 1:
            return [audio.samples.copy()], {
                "backend": "cluster-diarization",
                "confidence": float(est["confidence"]),
                "labels": ["Speaker 1"],
                "metrics": {
                    "estimated_speakers": est["n_speakers"],
                    "embedding": est["method"],
                    "silhouette_scores": est["scores"],
                    "note": "single speaker detected" if k == 1 else "not enough speech windows",
                },
            }

        feats = ecapa_embeddings(audio, fine_windows)
        embedding_name = "ecapa-tdnn"
        if feats is None or len(feats) != len(fine_windows):
            feats = window_features(audio, fine_windows)
            embedding_name = "mfcc-numpy"

        k = min(k, max(1, len(feats)))
        labels, centres = kmeans(feats, k, seed=7)
        labels = _median_filter_labels(labels, size=5)

        # --- windows -> per speaker segments ------------------------------ #
        per_speaker: List[List[Tuple[float, float]]] = [[] for _ in range(k)]
        for (start, end), lab in zip(fine_windows, labels):
            per_speaker[int(lab)].append((float(start), float(end)))

        sources: List[np.ndarray] = []
        seg_lists: List[List[List[float]]] = []
        kept_labels: List[str] = []
        n = audio.n_samples
        for i in range(k):
            merged = _merge_segments(per_speaker[i], gap=0.35)
            merged = [s for s in merged if (s[1] - s[0]) >= 0.35]
            if not merged:
                continue
            mask = segments_to_mask(merged, n, audio.sr, fade_ms=25.0)
            sources.append((audio.samples * mask).astype(np.float32))
            seg_lists.append(merged)
            kept_labels.append("Speaker %d" % (len(kept_labels) + 1))

        if not sources:  # degenerate: give back the original
            return [audio.samples.copy()], {
                "backend": "cluster-diarization",
                "confidence": 0.2,
                "metrics": {"note": "clustering produced no usable segments"},
            }

        # order speakers by first appearance -> matches how a human would label
        order = sorted(range(len(sources)), key=lambda i: seg_lists[i][0][0])
        sources = [sources[i] for i in order]
        seg_lists = [seg_lists[i] for i in order]

        talk = [sum(e - s for s, e in segs) for segs in seg_lists]
        return sources, {
            "backend": "cluster-diarization",
            "confidence": float(est["confidence"]),
            "labels": ["Speaker %d" % (i + 1) for i in range(len(sources))],
            "segments": seg_lists,
            "metrics": {
                "estimated_speakers": int(est["n_speakers"]),
                "requested_speakers": num_speakers,
                "embedding": embedding_name,
                "silhouette_scores": est["scores"],
                "eigengap_k": est.get("eigengap_k"),
                "n_windows": len(fine_windows),
                "talk_time_seconds": [round(float(t), 2) for t in talk],
                "turns_per_speaker": [len(s) for s in seg_lists],
            },
        }
