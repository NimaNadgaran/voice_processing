"""Sample-rate independent minimum-statistics noise tracking."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def spectral_sizes(sr, n_fft=None, hop=None):
    # Preserve the 64 ms window / 16 ms hop at every input sample rate.
    n_fft = int(n_fft) if n_fft is not None else max(32, 2 * round(sr * .032))
    hop = int(hop) if hop is not None else max(1, n_fft // 4)
    if n_fft < 2 or n_fft % 2 or not 0 < hop <= n_fft // 2:
        raise ValueError("require an even n_fft and 0 < hop <= n_fft / 2")
    return n_fft, hop


def noise_profile(power, sr, hop, factor=2.):
    """Local minima of averaged power adapt to noise changes without hard gates.

    Averaging first avoids treating random near-zero FFT bins as the noise
    floor. The correction compensates for the downward bias of the minima.
    """
    smooth = uniform_filter1d(power, size=5, axis=1, mode="nearest")
    size = max(3, round(.8 * sr / hop))
    floor = minimum_filter1d(smooth, size=size, axis=1, mode="nearest")
    return smooth, np.maximum(factor * floor, 1e-12)
