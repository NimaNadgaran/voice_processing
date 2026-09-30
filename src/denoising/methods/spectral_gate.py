"""Method 1 -- Spectral gating (a.k.a. "noise print" subtraction).

Backends
--------
1. ``noisereduce`` (pip install noisereduce)  -- the well known implementation
   with stationary / non-stationary modes.
2. **built-in numpy fallback** -- a from-scratch spectral gate so this method is
   *always* available, even on a bare ``pip install -r requirements.txt``.

How it works
------------
Build a per-frequency noise profile from the quietest frames of the clip, put a
threshold a few dB above it, and attenuate every time-frequency bin that falls
below the threshold.  The mask is smoothed in both time and frequency to avoid
the "musical noise" chirps that naive spectral subtraction produces.

Strengths : instant, CPU only, no model download, keeps voice timbre intact.
Weakness  : assumes the noise is roughly stationary (fans, hiss, hum, traffic).
            It will not remove another person talking.
"""

from __future__ import annotations

import os

from typing import Any, Dict, Tuple

import numpy as np

from ...core.dsp import EPS, istft, magphase, stft
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser
from ..noise import noise_profile, spectral_sizes


def _smooth2d(mask: np.ndarray, f_size: int = 4, t_size: int = 4) -> np.ndarray:
    """Separable box blur over the (freq, time) mask."""
    from scipy.ndimage import uniform_filter1d
    out = mask.astype(np.float32)
    if f_size > 1:
        out = uniform_filter1d(out, size=f_size, axis=0, mode="nearest")
    if t_size > 1:
        out = uniform_filter1d(out, size=t_size, axis=1, mode="nearest")
    return out.astype(np.float32)


def builtin_spectral_gate(
    x: np.ndarray,
    sr: int,
    n_fft: int | None = None,
    hop: int | None = None,
    n_std_thresh: float = 1.5,
    prop_decrease: float = 0.90,
    noise_percentile: float = 12.0,
    freq_smooth: int = 1,
    time_smooth: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Adaptive spectral subtraction. Returns (audio, info).

    noise_percentile is retained for compatibility with the old global noise
    profile; the adaptive tracker does not use it.
    """
    n_fft, hop = spectral_sizes(sr, n_fft, hop)
    if not 0 <= prop_decrease <= 1 or n_std_thresh <= 0:
        raise ValueError("require 0 <= prop_decrease <= 1 and positive threshold")
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32), {"backend": "builtin-adaptive", "frames": 0}
    spec = stft(x, n_fft=n_fft, hop=hop)
    mag, phase = magphase(spec)
    # Continuous power subtraction keeps consonants and avoids binary-mask
    # chirps. The local profile follows a fan/traffic level changing mid-clip.
    power, noise = noise_profile(mag**2, sr, hop, factor=2.0 * n_std_thresh / 1.5)
    mask = np.sqrt(np.clip(1 - noise / np.maximum(power, EPS), 0, 1))
    mask = _smooth2d(mask, freq_smooth, time_smooth)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask * prop_decrease + (1.0 - prop_decrease)

    out = istft(mag * mask * phase, hop=hop, length=len(x))
    info = {
        "backend": "builtin-adaptive",
        "noise_profile_db": round(float(10 * np.log10(np.maximum(noise.mean(), EPS))), 2),
        "noise_tracking": "local-minimum-statistics",
        "mean_gate_open": round(float(mask.mean()), 4),
        "n_fft": n_fft,
        "hop": hop,
    }
    return out.astype(np.float32), info


@register_denoiser
class SpectralGateDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="spectral_gate",
        name="Spectral Gating",
        kind="denoise",
        family="dsp",
        description=(
            "Adaptive soft spectral subtraction. Tracks the hiss/fan/hum profile "
            "locally and attenuates it. Instant, CPU-only, no downloads."
        ),
        speed="realtime",
        quality=3,
        needs_gpu=False,
        offline=True,
        pip=[],  # built-in fallback => always available
        install_hint="",  # nothing to install: the built-in numpy gate is the default
        notes="Best on stationary noise. Won't remove other voices or sudden bangs. "
              "Set SPECTRAL_GATE_BACKEND=noisereduce to use noisereduce instead "
              "(optional alternative, not automatically selected).",
    )

    def check_available(self) -> Tuple[bool, str]:
        return True, ""  # the numpy fallback is always there

    def _denoise(self, audio: AudioBuffer):
        x, sr = audio.samples, audio.sr

        # Keep optional noisereduce explicitly opt-in: installing a dependency
        # must not silently change the tuned default (see AUDIO_VALIDATION.md).
        if os.environ.get("SPECTRAL_GATE_BACKEND", "builtin").lower() == "noisereduce":
            if module_available("noisereduce"):
                try:
                    import noisereduce as nr

                    out = nr.reduce_noise(
                        y=x,
                        sr=sr,
                        stationary=True,
                        prop_decrease=0.5,
                        n_fft=spectral_sizes(sr)[0],
                        hop_length=spectral_sizes(sr)[1],
                        time_mask_smooth_ms=64,
                        freq_mask_smooth_hz=500,
                    )
                    return AudioBuffer(np.asarray(out, dtype=np.float32), sr), {
                        "backend": "noisereduce",
                        "mode": "stationary",
                    }
                except Exception as exc:  # fall through to the built-in gate
                    out, info = builtin_spectral_gate(x, sr)
                    info["backend"] = "builtin-fallback"
                    info["noisereduce_error"] = str(exc)
                    return AudioBuffer(out, sr), info

        out, info = builtin_spectral_gate(x, sr)
        return AudioBuffer(out, sr), info
