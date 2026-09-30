"""Method 6 -- NVIDIA NeMo MSDD neural diarization (advanced / heavy).

Install
-------
    pip install torch torchaudio
    pip install "nemo_toolkit[asr]"

MSDD = Multi-scale Diarization Decoder.  It runs speaker embedding extraction at
several window scales (1.5 s / 1.0 s / 0.5 s ...), fuses them with learned
scale weights, and decodes pairwise speaker presence -- which lets it handle
**overlapping speech** far better than plain clustering.  NVIDIA's telephonic and
meeting checkpoints are strong and free.

Checkpoints (``NEMO_DIAR_MODEL``):
* ``diar_msdd_telephonic`` -- 8 kHz phone-style audio (default)
* ``diar_msdd_meeting``    -- far-field meeting rooms

This backend is the heaviest install in the project (NeMo pulls in a large
dependency tree). It is listed as *experimental*: the NeMo diarizer API has
changed across releases, so we probe two call styles and fall back to the RTTM
file it writes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.audio_io import resample, save_audio
from ...core.dsp import segments_to_mask
from ...core.registry import register_separator
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import CACHE_DIR, module_available, new_id
from ..base import BaseSeparator


@register_separator
class NemoMSDDSeparator(BaseSeparator):
    info = MethodInfo(
        key="nemo_msdd",
        name="NeMo MSDD (NVIDIA)",
        kind="separate",
        family="deep-pretrained",
        description=(
            "Multi-scale Diarization Decoder: fuses speaker embeddings at several "
            "window scales and decodes overlapping speakers pairwise. Very strong on "
            "phone calls and meeting rooms."
        ),
        speed="slow",
        quality=5,
        needs_gpu=False,
        offline=True,
        pip=["torch", "nemo"],
        install_hint='pip install torch torchaudio && pip install "nemo_toolkit[asr]"',
        notes="EXPERIMENTAL: huge install, API varies by NeMo version. Any number of speakers.",
        max_speakers=None,
    )
    target_sr = 16000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("torch"):
            return False, "missing python package(s): torch"
        if not module_available("nemo"):
            return False, 'missing python package(s): nemo_toolkit[asr]'
        return True, ""

    def load(self) -> None:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer  # type: ignore

        name = os.environ.get("NEMO_DIAR_MODEL", "diar_msdd_telephonic")
        self._model = NeuralDiarizer.from_pretrained(model_name=name)
        self._model_name = name
        self._loaded = True

    def _separate(self, audio: AudioBuffer, num_speakers: Optional[int]):
        work_dir = CACHE_DIR / new_id("nemo")
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_path = work_dir / "input.wav"
        save_audio(wav_path, resample(audio, 16000))

        segments = self._run(wav_path, work_dir, num_speakers)
        if not segments:
            raise RuntimeError("NeMo produced no diarization output (check the NeMo version)")

        order = sorted(segments, key=lambda s: segments[s][0][0])
        sources: List[np.ndarray] = []
        seg_lists: List[List[List[float]]] = []
        labels: List[str] = []
        for i, speaker in enumerate(order):
            segs = sorted(segments[speaker])
            mask = segments_to_mask(segs, audio.n_samples, audio.sr, fade_ms=25.0)
            sources.append((audio.samples * mask).astype(np.float32))
            seg_lists.append(segs)
            labels.append("Speaker %d (%s)" % (i + 1, speaker))

        return sources, {
            "backend": "nemo-msdd",
            "confidence": 0.85,
            "labels": labels,
            "segments": seg_lists,
            "metrics": {
                "model": self._model_name,
                "detected_speakers": len(order),
                "requested_speakers": num_speakers,
                "talk_time_seconds": [round(sum(e - s for s, e in segs), 2) for segs in seg_lists],
            },
        }

    # ------------------------------------------------------------------ #
    def _run(self, wav_path: Path, work_dir: Path, num_speakers: Optional[int]) -> Dict[str, List[List[float]]]:
        # style A: callable diarizer returning a pyannote-style Annotation
        try:
            kwargs = {"audio_filepath": str(wav_path)}
            if num_speakers:
                kwargs["num_speakers"] = int(num_speakers)
            annotation = self._model(**kwargs)
            out: Dict[str, List[List[float]]] = {}
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                out.setdefault(str(speaker), []).append([float(turn.start), float(turn.end)])
            if out:
                return out
        except Exception:
            pass

        # style B: manifest + diarize() writing an RTTM into work_dir
        try:
            import json

            manifest = work_dir / "manifest.json"
            entry = {
                "audio_filepath": str(wav_path),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": int(num_speakers) if num_speakers else None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            cfg = getattr(self._model, "cfg", None)
            if cfg is not None:
                cfg.diarizer.manifest_filepath = str(manifest)
                cfg.diarizer.out_dir = str(work_dir)
            self._model.diarize()
        except Exception:
            pass

        return self._parse_rttm(work_dir)

    @staticmethod
    def _parse_rttm(work_dir: Path) -> Dict[str, List[List[float]]]:
        out: Dict[str, List[List[float]]] = {}
        for rttm in work_dir.rglob("*.rttm"):
            for line in rttm.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                start, dur, speaker = float(parts[3]), float(parts[4]), parts[7]
                out.setdefault(speaker, []).append([start, start + dur])
        return out
