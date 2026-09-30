"""Bounded neural inference with context on both sides and smooth joins."""
from __future__ import annotations

import numpy as np
from ..core.audio_io import match_length


def chunked_enhance(fn, x, sr, chunk_s=10., overlap_s=.5, context_s=.25):
    x = np.asarray(x, dtype=np.float32)
    if sr <= 0 or not 0 < overlap_s < chunk_s or context_s < 0:
        raise ValueError("require positive sr, 0 < overlap_s < chunk_s, and nonnegative context")
    if not len(x):
        return x.copy()
    chunk, overlap, context = int(chunk_s * sr), max(1, int(overlap_s * sr)), int(context_s * sr)
    result, weight = np.zeros_like(x), np.zeros_like(x)
    for start in range(0, len(x), max(1, chunk - overlap)):
        end = min(start + chunk, len(x))
        left, right = max(0, start - context), min(len(x), end + context)
        output = match_length(fn(x[left:right]), right - left)[start - left:end - left]
        win = np.ones(end - start, dtype=np.float32)
        fade_len = min(overlap, len(win))
        ramp = np.linspace(0, 1, fade_len + 2, dtype=np.float32)[1:-1]
        if start:
            win[:fade_len] *= ramp
        if end < len(x):
            win[-fade_len:] *= ramp[::-1]
        result[start:end] += output * win
        weight[start:end] += win
        if end == len(x):
            break
    return result / np.maximum(weight, 1e-8)
