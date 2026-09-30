"""Method 4 -- version-matched pyannote.audio diarization.

Install
-------
    pip install torch torchaudio
    pip install pyannote.audio

Then, once, in a browser:
1. On pyannote.audio 4.x accept the conditions for
   huggingface.co/pyannote/speaker-diarization-community-1. On 3.x accept
   speaker-diarization-3.1 and segmentation-3.0,
2. create a read token and ``set HF_TOKEN=hf_xxx``.

Why it is the best default for meetings
---------------------------------------
The 3.1 pipeline runs a powerful end-to-end segmentation model (which *does*
detect overlapped speech) followed by agglomerative clustering of speaker
embeddings.  It handles **any** number of speakers, discovers the count on its
own, and is the strongest freely available system for real recordings
(~10-20% DER on DIHARD-style data).

You can also pin the count: the pipeline accepts ``num_speakers`` or
``min_speakers``/``max_speakers``, which the UI passes through when you set it.

Note it is diarization, so overlapped regions are attributed to *all* active
speakers rather than being acoustically unmixed.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Tuple

import numpy as np

from ...core.dsp import segments_to_mask
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseSeparator


def _token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "PYANNOTE_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def pipeline_checkpoint():
    override = os.environ.get('PYANNOTE_PIPELINE', '').strip()
    if override:
        return override
    try:
        major = int(version('pyannote.audio').split('.')[0])
    except (PackageNotFoundError, ValueError):
        major = 3
    # 4.x changed pipeline internals as well as the auth keyword. Pair it with
    # its own config instead of accidentally mixing the 3.1 and 4.x pipelines.
    return 'pyannote/speaker-diarization-community-1' if major >= 4 else 'pyannote/speaker-diarization-3.1'


@register_separator
class PyannoteDiarizationSeparator(BaseSeparator):
    info = MethodInfo(
        key="pyannote",
        name="pyannote.audio diarization",
        kind="separate",
        family="deep-pretrained",
        description=(
            "The reference open-source diarization pipeline: neural segmentation "
            "with overlap detection + embedding clustering. Finds ANY number of "
            "speakers by itself and is the strongest option on real meetings."
        ),
        speed="medium",
        quality=5,
        needs_gpu=False,
        offline=True,
        pip=["torch", "pyannote.audio"],
        install_hint=(
            "pip install torch torchaudio pyannote.audio, accept the model licence on "
            "huggingface.co/pyannote/speaker-diarization-3.1, then set HF_TOKEN=hf_xxx"
        ),
        notes="Weights download once (free). Attributes overlaps to all active speakers.",
        max_speakers=None,
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("pyannote.audio"):
            return False, "missing python package(s): pyannote.audio"
        if not _token():
            return False, "set HF_TOKEN and accept the pyannote model licence (both free)"
        return True, ""

    def describe(self) -> MethodInfo:
        # Once the wheel is in, the only thing left is the token + licence --
        # printing the full pip line again just buries the step that is missing.
        info = super().describe()
        if module_available("pyannote.audio") and module_available("torch"):
            info.install_hint = (
                "accept the model licence at huggingface.co/" + pipeline_checkpoint() +
                ", then set HF_TOKEN=hf_xxx for that account"
            )
        return info

    def load(self) -> None:
        import torch  # type: ignore
        from pyannote.audio import Pipeline  # type: ignore

        checkpoint = pipeline_checkpoint()
        self._checkpoint = checkpoint
        # pyannote.audio 4.x renamed the auth argument use_auth_token -> token.
        try:
            pipeline = Pipeline.from_pretrained(checkpoint, token=_token())
        except TypeError:
            pipeline = Pipeline.from_pretrained(checkpoint, use_auth_token=_token())
        if pipeline is None:
            raise RuntimeError(
                f"pyannote could not load '{checkpoint}' -- accept the model licence at "
                f"huggingface.co/{checkpoint} with the same account as your HF_TOKEN"
            )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        self._model = pipeline
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        import torch  # type: ignore

        wav = torch.from_numpy(np.ascontiguousarray(audio.samples)).float().unsqueeze(0)
        payload = {"waveform": wav, "sample_rate": audio.sr}

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)

        annotation = self._model(payload, **kwargs)
        annotation = getattr(annotation, "speaker_diarization", annotation)

        per_speaker: dict = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start, end = max(0., float(turn.start)), min(audio.duration, float(turn.end))
            if end > start:
                per_speaker.setdefault(speaker, []).append([start, end])

        if not per_speaker:
            return [audio.samples.copy()], {
                "backend": "pyannote",
                "confidence": 0.3,
                "metrics": {"note": "pipeline found no speech"},
            }

        order = sorted(per_speaker, key=lambda s: per_speaker[s][0][0])
        sources: List[np.ndarray] = []
        seg_lists: List[List[List[float]]] = []
        labels: List[str] = []
        for i, speaker in enumerate(order):
            segs = sorted(per_speaker[speaker])
            mask = segments_to_mask(segs, audio.n_samples, audio.sr, fade_ms=25.0)
            sources.append((audio.samples * mask).astype(np.float32))
            seg_lists.append(segs)
            labels.append("Speaker %d (%s)" % (i + 1, speaker))

        from ..diarization import merge_turns, overlap_ratio
        seg_lists = [merge_turns(segs, audio.duration) for segs in seg_lists]
        talk = [sum(e - s for s, e in segs) for segs in seg_lists]
        overlap = overlap_ratio(seg_lists, audio.duration)

        return sources, {
            "backend": "pyannote",
            "confidence": 0.9,
            "labels": labels,
            "segments": seg_lists,
            "metrics": {
                "detected_speakers": len(order),
                "requested_speakers": num_speakers,
                "talk_time_seconds": [round(t, 2) for t in talk],
                "turns_per_speaker": [len(s) for s in seg_lists],
                "estimated_overlap_ratio": round(float(overlap), 4),
                "pipeline": getattr(self, '_checkpoint', pipeline_checkpoint()),
            },
        }
