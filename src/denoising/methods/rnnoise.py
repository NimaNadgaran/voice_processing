"""Method 6 -- RNNoise (Xiph), the 85 kB GRU that ships in WebRTC/OBS.

Install
-------
    pip install pyrnnoise

The tiniest useful neural denoiser: a hybrid DSP + GRU model that predicts band
gains every 10 ms.  It costs almost nothing (well under 1% of one CPU core in
real time), needs no torch at all, and is the right choice for long recordings
or low-power machines.  Quality is below DeepFilterNet but clearly above a
plain gate on steady noise, and it also returns a per-frame *speech probability*
which we surface as a VAD curve.

Reference: Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
Speech Enhancement", MMSP 2018.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser

FRAME = 480  # 10 ms at 48 kHz -- RNNoise's native frame size
ALGORITHMIC_DELAY = 2 * FRAME  # librnnoise returns frame N-2; see _denoise_lowlevel


@register_denoiser
class RNNoiseDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="rnnoise",
        name="RNNoise (Xiph)",
        kind="denoise",
        family="deep-pretrained",
        description=(
            "The 85 kB GRU denoiser used by WebRTC, OBS and Mumble. Hybrid DSP + "
            "neural band gains at 10 ms latency, no torch required. Also gives a "
            "free voice-activity probability per frame."
        ),
        speed="realtime",
        quality=3,
        needs_gpu=False,
        offline=True,
        pip=["pyrnnoise"],
        install_hint="pip install pyrnnoise",
        notes="Runs at 48 kHz, ~0.005x real time. Great for very long files.",
    )
    target_sr = 48000
    restore_sr = True

    def check_available(self) -> Tuple[bool, str]:
        if not module_available("pyrnnoise"):
            return False, "missing python package(s): pyrnnoise"
        return True, ""

    def load(self) -> None:
        # Preferred: the thin ctypes binding around librnnoise.  It takes one
        # 480-sample int16 frame and hands back the frame plus a speech
        # probability -- no resampling graph, no audiolab, and it has not
        # changed shape across pyrnnoise releases.  We already resample to
        # 48 kHz ourselves (target_sr), so the wrapper buys us nothing.
        self._lowlevel = None
        try:
            from pyrnnoise.rnnoise import create, destroy, process_frame  # type: ignore

            self._lowlevel = (create, destroy, process_frame)
        except Exception:
            pass

        if self._lowlevel is None:
            from pyrnnoise import RNNoise  # type: ignore

            self._factory = RNNoise
        self._loaded = True

    def _denoise(self, audio: AudioBuffer):
        if self._lowlevel is not None:
            return self._denoise_lowlevel(audio)
        return self._denoise_wrapper(audio)

    def _denoise_lowlevel(self, audio: AudioBuffer):
        create, destroy, process_frame = self._lowlevel

        pcm = np.clip(audio.samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        pad = (-len(pcm16)) % FRAME
        if pad:
            pcm16 = np.pad(pcm16, (0, pad))

        state = create()
        try:
            out_chunks = []
            probs = []
            for start in range(0, len(pcm16), FRAME):
                frame, prob = process_frame(state, pcm16[start: start + FRAME])
                out_chunks.append(np.asarray(frame, dtype=np.int16).reshape(-1))
                probs.append(float(np.mean(np.asarray(prob, dtype=np.float64))))
        finally:
            destroy(state)

        arr = np.concatenate(out_chunks).astype(np.float32) / 32767.0

        # librnnoise runs two frames behind its input: the sample you get back
        # for frame N is really frame N-2.  Measured at exactly 960 samples
        # (2 x 480 = 20 ms at 48 kHz) and constant across runs.  Left in, that
        # 20 ms shift wrecks every correlation-based metric (SI-SDR came out at
        # -23 dB on real speech that was actually cleaned well) and pushes the
        # speaker tracks off the original timeline.  Drop the priming samples
        # and pad the tail so length is preserved.
        if len(arr) > ALGORITHMIC_DELAY:
            arr = np.concatenate(
                [arr[ALGORITHMIC_DELAY:], np.zeros(ALGORITHMIC_DELAY, dtype=np.float32)]
            )
        if pad:
            arr = arr[: len(arr) - pad]

        info = {
            "backend": "rnnoise",
            "frames": len(out_chunks),
            "delay_compensated_samples": ALGORITHMIC_DELAY,
        }
        if probs:
            info["mean_speech_prob"] = round(float(np.mean(probs)), 4)
            info["voiced_frame_ratio"] = round(float(np.mean(np.array(probs) > 0.5)), 4)
        return AudioBuffer(arr, audio.sr), info

    def _denoise_wrapper(self, audio: AudioBuffer):
        denoiser = self._factory(self.target_sr)
        pcm = np.clip(audio.samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)

        pad = (-len(pcm16)) % FRAME
        if pad:
            pcm16 = np.pad(pcm16, (0, pad))

        out_chunks = []
        probs = []
        for start in range(0, len(pcm16), FRAME):
            frame = pcm16[start: start + FRAME]
            processed = self._process_frame(denoiser, frame)
            if processed is None:
                continue
            prob, data = processed
            if prob is not None:
                probs.append(float(prob))
            out_chunks.append(np.asarray(data, dtype=np.int16).reshape(-1))

        if not out_chunks:
            raise RuntimeError("pyrnnoise returned no frames -- unexpected API shape")

        arr = np.concatenate(out_chunks).astype(np.float32) / 32767.0
        if pad:
            arr = arr[: len(arr) - pad] if len(arr) > pad else arr

        info = {"backend": "rnnoise", "frames": len(out_chunks)}
        if probs:
            info["mean_speech_prob"] = round(float(np.mean(probs)), 4)
            info["voiced_frame_ratio"] = round(float(np.mean(np.array(probs) > 0.5)), 4)
        return AudioBuffer(arr, audio.sr), info

    # pyrnnoise has changed its surface a couple of times -- probe politely
    @staticmethod
    def _process_frame(denoiser, frame: np.ndarray):
        if hasattr(denoiser, "process_frame"):
            res = denoiser.process_frame(frame)
            if isinstance(res, tuple):
                return res[0], res[1]
            return None, res
        if hasattr(denoiser, "process_chunk"):
            last = None
            for item in denoiser.process_chunk(frame):
                last = item
            if last is None:
                return None
            if isinstance(last, tuple):
                return last[0], last[1]
            return None, last
        if hasattr(denoiser, "denoise_frame"):
            res = denoiser.denoise_frame(frame)
            # 0.4.x returns (samples, speech_prob) -- our caller wants it the
            # other way round.
            return (res[1], res[0]) if isinstance(res, tuple) else (None, res)
        if hasattr(denoiser, "denoise_chunk"):
            # a generator of (samples, speech_prob); drain it and keep the last
            last = None
            for item in denoiser.denoise_chunk(frame):
                last = item
            if last is None:
                return None
            return (last[1], last[0]) if isinstance(last, tuple) else (None, last)
        raise RuntimeError("unsupported pyrnnoise version: no known process method")
