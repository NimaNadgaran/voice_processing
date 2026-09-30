"""Long-file support for fixed-length neural separators.

Transformer separators (SepFormer & friends) have quadratic memory in the input
length, so a 20 minute meeting cannot be fed in one go.  We cut the signal into
overlapping blocks, separate each block, and then stitch the blocks back
together.

The subtle part is **permutation stitching**: a separator has no notion of
identity, so block 1 may output (Alice, Bob) while block 2 outputs (Bob, Alice).
Before overlap-adding a block we test every permutation against the tail of the
already-assembled output and keep the one with the highest correlation in the
overlap region.  Without this step the speakers swap files every few seconds.
"""

from __future__ import annotations

import itertools
from typing import Callable, List

import numpy as np

SeparateFn = Callable[[np.ndarray], List[np.ndarray]]


def _best_permutation(prev_tail: np.ndarray, cand: np.ndarray) -> tuple:
    """prev_tail/cand: (n_src, overlap). Returns the permutation of `cand` rows."""
    n = prev_tail.shape[0]
    best, best_score = tuple(range(n)), -np.inf
    a = prev_tail - prev_tail.mean(axis=1, keepdims=True)
    b = cand - cand.mean(axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    corr = a @ b.T  # (n_prev, n_cand)
    for perm in itertools.permutations(range(n)):
        score = float(sum(corr[i, perm[i]] for i in range(n)))
        if score > best_score:
            best, best_score = perm, score
    return best


def chunked_separate(
    fn: SeparateFn,
    x: np.ndarray,
    sr: int,
    chunk_s: float = 10.0,
    overlap_s: float = 1.0,
    max_direct_s: float = 15.0,
) -> List[np.ndarray]:
    """Run ``fn`` over long audio and stitch the sources with permutation fixing."""
    x = np.asarray(x, dtype=np.float32)
    if sr <= 0 or chunk_s <= 0 or not 0 < overlap_s < chunk_s or max_direct_s < 0:
        raise ValueError("require positive sample rate and 0 < overlap_s < chunk_s")
    if len(x) <= int(max_direct_s * sr):
        from ..core.audio_io import match_length
        return [match_length(s, len(x)) for s in fn(x)]

    chunk = int(chunk_s * sr)
    overlap = max(int(overlap_s * sr), 1)
    step = max(chunk - overlap, 1)

    out: np.ndarray | None = None
    weight: np.ndarray | None = None
    fade = np.hanning(2 * overlap).astype(np.float32)

    pos = 0
    while pos < len(x):
        end = min(pos + chunk, len(x))
        block = x[pos:end]
        # Full context on the final call also protects convolutional/attention
        # backends from a tiny tail shorter than their encoder kernel.
        if len(block) < chunk:
            block = np.pad(block, (0, chunk - len(block)))
        sources = [np.asarray(s, dtype=np.float32).reshape(-1)[: end - pos] for s in fn(block)]
        n_src = len(sources)
        if not n_src:
            raise RuntimeError("separator returned no sources for a chunk")
        stack = np.stack([np.pad(s, (0, end - pos - len(s))) for s in sources])

        if out is None:
            out = np.zeros((n_src, len(x)), dtype=np.float32)
            weight = np.zeros(len(x), dtype=np.float32)
        elif out.shape[0] != n_src:  # model changed its mind about the count
            raise RuntimeError("separator changed source count between chunks")

        if pos > 0 and overlap > 1:
            actual = min(overlap, end - pos)
            prev_tail = out[:n_src, pos: pos + actual] / np.maximum(weight[pos: pos + actual], 1e-6)
            perm = _best_permutation(prev_tail, stack[:, :actual])
            stack = stack[list(perm)]

        win = np.ones(stack.shape[1], dtype=np.float32)
        if pos > 0:
            size = min(overlap, len(win))
            win[:size] = fade[:size]
        if end < len(x) and len(win) > overlap:
            win[-overlap:] = fade[overlap:]

        out[:n_src, pos:end] += stack * win[None, :]
        weight[pos:end] += win
        if end >= len(x):
            break
        pos += step

    if out is None:
        return [np.asarray(s, dtype=np.float32) for s in fn(x)]
    out /= np.maximum(weight, 1e-6)[None, :]
    return [row.astype(np.float32) for row in out]
