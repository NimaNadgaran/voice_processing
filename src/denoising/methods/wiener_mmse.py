"""Method 2 -- MMSE-LSA / Wiener statistical denoiser (Ephraim & Malah).

Pure numpy (scipy only for the exponential integral, with a series fallback).
Always available, no downloads, no training.

Pipeline per STFT frame
-----------------------
1. Track the noise PSD recursively, gated by a speech-presence probability
   (an MCRA-flavoured update) so it keeps adapting to changing noise.
2. Estimate the *a priori* SNR ``xi`` with the decision-directed approach,
   which is what kills the musical-noise artefacts of plain subtraction.
3. Apply the MMSE log-spectral-amplitude gain
   ``G = xi/(1+xi) * exp(0.5 * E1(v))``, floored so the residual noise stays
   natural instead of turning into silence.

Strengths : very natural sounding, no "underwater" artefacts, real-time speed,
            adapts to slowly changing noise.
Weakness  : ~6-12 dB of suppression only; it will not touch babble or music.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...core.dsp import EPS, istft, magphase, stft
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import module_available
from ..base import BaseDenoiser


def _exp1(v: np.ndarray) -> np.ndarray:
    """E1(v) = int_v^inf e^-t / t dt."""
    if module_available("scipy"):
        from scipy.special import exp1  # type: ignore

        return np.asarray(exp1(np.maximum(v, 1e-8)), dtype=np.float32)
    # Abramowitz & Stegun 5.1.53 / 5.1.56 rational approximations
    v = np.maximum(v, 1e-8)
    small = v <= 1.0
    out = np.empty_like(v, dtype=np.float32)
    a = [-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857]
    vs = v[small]
    out[small] = (a[0] + vs * (a[1] + vs * (a[2] + vs * (a[3] + vs * (a[4] + vs * a[5]))))
                  - np.log(np.maximum(vs, 1e-12)))
    vb = v[~small]
    num = vb * vb + 2.334733 * vb + 0.250621
    den = vb * vb + 3.330657 * vb + 1.681534
    out[~small] = np.exp(-vb) / vb * (num / den)
    return out.astype(np.float32)


def mmse_lsa(
    x: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    alpha_dd: float = 0.92,
    alpha_noise: float = 0.95,
    gain_floor_db: float = -18.0,
    init_frames: int = 8,
) -> Tuple[np.ndarray, dict]:
    """Defaults were chosen by measurement, not by tradition.

    Swept against ground truth on the bundled demo (see scripts/benchmark.py):
    the textbook ``n_fft=512, alpha_dd=0.98`` scored **-0.9 dB** SI-SDR -- worse
    than doing nothing -- while ``n_fft=1024, alpha_dd=0.92`` scored **+5.1 dB**.
    A 64 ms window resolves the pitch harmonics that a 32 ms one smears, and the
    lighter decision-directed smoothing stops the gain lagging behind speech
    onsets.
    """
    spec = stft(x, n_fft=n_fft, hop=hop)
    mag, phase = magphase(spec)
    power = mag**2
    n_bins, n_frames = power.shape
    if n_frames == 0:
        return x.astype(np.float32), {"backend": "mmse-lsa", "frames": 0}

    # Initialise the noise floor from a low percentile over the WHOLE file, not
    # from the opening frames. Assuming a recording starts with silence is wrong
    # far more often than it is right -- when it starts mid-sentence that seeds
    # the noise estimate with speech power and the estimator then suppresses the
    # voice itself (measured: -9 dB SI-SDR, i.e. worse than doing nothing).
    # The leading frames are still used, but only if they are quieter.
    noise_psd = np.percentile(power, 10, axis=1).astype(np.float64) + EPS
    lead = np.mean(power[:, : max(1, min(init_frames, n_frames))], axis=1) + EPS
    noise_psd = np.minimum(noise_psd, lead)

    gain_floor = 10.0 ** (gain_floor_db / 20.0)
    gains = np.empty_like(power, dtype=np.float32)
    prev_clean = np.maximum(power[:, 0] - noise_psd, EPS)
    speech_frames = 0

    for t in range(n_frames):
        p = power[:, t]
        gamma = np.minimum(p / noise_psd, 1e4)  # a posteriori SNR
        xi = alpha_dd * (prev_clean / noise_psd) + (1.0 - alpha_dd) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-4)

        v = (xi / (1.0 + xi)) * gamma
        g = (xi / (1.0 + xi)) * np.exp(0.5 * _exp1(v))
        g = np.clip(g, gain_floor, 1.0)
        gains[:, t] = g

        clean = (g * mag[:, t]) ** 2
        prev_clean = clean

        # --- speech presence probability, PER FREQUENCY BIN ------------------
        # Sohn's likelihood ratio. It must stay per-bin: averaging it across the
        # spectrum is dominated by the many bins that hold no speech even during
        # a vowel, so the frame reads as "silence", the noise estimate absorbs
        # the voice, and the estimator then suppresses the speech it is meant to
        # keep. Freezing the noise update per bin is the standard MCRA fix.
        lr = np.exp(np.clip(v - np.log1p(xi), -30, 30))
        p_speech_bin = lr / (1.0 + lr)

        # Sohn's frame-level statistic (geometric mean) purely for reporting
        if float(np.mean(np.log(np.maximum(lr, 1e-12)))) > 0.0:
            speech_frames += 1

        smoothing = alpha_noise + (1.0 - alpha_noise) * p_speech_bin  # per bin
        noise_psd = smoothing * noise_psd + (1.0 - smoothing) * p
        noise_psd = np.maximum(noise_psd, EPS)

    out = istft(mag * gains * phase, hop=hop, length=len(x))
    info = {
        "backend": "mmse-lsa",
        "frames": int(n_frames),
        "mean_gain_db": round(float(20 * np.log10(np.maximum(gains.mean(), EPS))), 2),
        "speech_frame_ratio": round(speech_frames / float(n_frames), 4),
        "final_noise_floor_db": round(float(10 * np.log10(np.maximum(noise_psd.mean(), EPS))), 2),
    }
    return out.astype(np.float32), info


@register_denoiser
class WienerMMSEDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="wiener_mmse",
        name="MMSE-LSA (Ephraim-Malah)",
        kind="denoise",
        family="dsp",
        description=(
            "Statistical minimum-mean-square-error log-spectral-amplitude estimator "
            "with decision-directed SNR tracking. The gold standard classic - very "
            "natural, artefact free, real-time."
        ),
        speed="realtime",
        quality=3,
        needs_gpu=False,
        offline=True,
        pip=[],
        install_hint="no install needed (numpy + scipy)",
        notes="Adaptive noise tracking: handles noise that changes slowly over time.",
    )

    def check_available(self) -> Tuple[bool, str]:
        return True, ""

    def _denoise(self, audio: AudioBuffer):
        out, info = mmse_lsa(audio.samples, audio.sr)
        return AudioBuffer(out, audio.sr), info
