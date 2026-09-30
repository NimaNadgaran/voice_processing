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


def _smooth2d(mask: np.ndarray, f_size: int = 4, t_size: int = 4) -> np.ndarray:
    """Separable box blur over the (freq, time) mask."""
    out = mask.astype(np.float32)
    if f_size > 1:
        k = np.ones(f_size, dtype=np.float32) / f_size
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
    if t_size > 1:
        k = np.ones(t_size, dtype=np.float32) / t_size
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out.astype(np.float32)


def builtin_spectral_gate(
    x: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    n_std_thresh: float = 1.5,
    prop_decrease: float = 0.95,
    noise_percentile: float = 12.0,
    freq_smooth: int = 4,
    time_smooth: int = 6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dependency-free spectral gate. Returns (audio, info)."""
    spec = stft(x, n_fft=n_fft, hop=hop)
    mag, phase = magphase(spec)
    log_mag = np.log(np.maximum(mag, EPS))

    # --- noise profile: the quietest `noise_percentile`% of frames per bin ---
    noise_mu = np.percentile(log_mag, noise_percentile, axis=1)
    noise_sd = np.std(log_mag[log_mag <= noise_mu[:, None]].reshape(-1)) if log_mag.size else 0.5
    noise_sd = float(noise_sd) if np.isfinite(noise_sd) and noise_sd > 0 else 0.5
    thresh = (noise_mu + n_std_thresh * noise_sd)[:, None]

    # --- soft mask + smoothing ------------------------------------------- #
    hard = (log_mag > thresh).astype(np.float32)
    mask = _smooth2d(hard, freq_smooth, time_smooth)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask * prop_decrease + (1.0 - prop_decrease)

    out = istft(mag * mask * phase, hop=hop, length=len(x))
    info = {
        "backend": "builtin-fallback",
        "noise_profile_db": round(float(np.mean(noise_mu) * 8.686), 2),
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
            "Classic noise-print gating. Learns the hiss/fan/hum profile from the "
            "quietest frames and subtracts it. Instant, CPU-only, no downloads."
        ),
        speed="realtime",
        quality=3,
        needs_gpu=False,
        offline=True,
        pip=[],  # built-in fallback => always available
        install_hint="",  # nothing to install: the built-in numpy gate is the default
        notes="Best on stationary noise. Won't remove other voices or sudden bangs. "
              "Set SPECTRAL_GATE_BACKEND=noisereduce to use noisereduce instead "
              "(measurably worse here -- see _denoise).",
    )

    def check_available(self) -> Tuple[bool, str]:
        return True, ""  # the numpy fallback is always there

    def _denoise(self, audio: AudioBuffer):
        x, sr = audio.samples, audio.sr

        # The built-in gate is the default on purpose.  Measured against known
        # ground truth (scripts/benchmark.py, synthetic 3spk @ 5 dB) it beats
        # `noisereduce` at every strength tried:
        #
        #     builtin                     purity 0.78  SI-SDRi +6.66  denoise +4.11
        #     noisereduce stat  p=0.50    purity 0.81  SI-SDRi +4.18  denoise +0.45
        #     noisereduce stat  p=0.95    purity 0.51  SI-SDRi +0.46  denoise -2.44
        #     noisereduce nstat p=0.95    purity 0.51  SI-SDRi +1.20  denoise -2.54
        #
        # noisereduce over-gates this material -- the last two rows are *worse
        # than not denoising at all*.  So it is opt-in rather than automatic;
        # merely having it installed must not silently degrade path1.
        if os.environ.get("SPECTRAL_GATE_BACKEND", "builtin").lower() == "noisereduce":
            if module_available("noisereduce"):
                try:
                    import noisereduce as nr

                    out = nr.reduce_noise(
                        y=x,
                        sr=sr,
                        stationary=True,
                        prop_decrease=0.5,   # gentlest setting measured; see above
                        n_fft=1024,
                        hop_length=256,
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
