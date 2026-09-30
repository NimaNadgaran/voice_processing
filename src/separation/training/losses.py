"""Permutation-invariant training (PIT) losses for separation.

The whole difficulty of source separation training is that the network has no
way to know *which* output should be which speaker -- "Alice, Bob" and
"Bob, Alice" are equally correct.  Training against a fixed order therefore
sends contradictory gradients and the model collapses to outputting the mixture
twice.

**uPIT** (utterance-level PIT, Kolbaek et al. 2017) fixes it: score every
permutation of outputs against the references and back-propagate only the best
one.  For ``n_src`` sources that is ``n_src!`` permutations -- trivial for 2-4.

``PITLossWrapper`` below is a drop-in replacement for Asteroid's class of the
same name, kept dependency-free.
"""

from __future__ import annotations

from itertools import permutations
from typing import Tuple

import torch
import torch.nn as nn

EPS = 1e-8


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (...) scale-invariant SNR in dB."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(dim=-1) + EPS) / (noise.pow(2).sum(dim=-1) + EPS))


def pairwise_si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """est/ref: (B, n_src, T) -> (B, n_src_est, n_src_ref) matrix of SI-SNR."""
    return si_snr(est.unsqueeze(2), ref.unsqueeze(1))


class PITLossWrapper(nn.Module):
    """Negative best-permutation SI-SNR (so lower loss = better separation)."""

    def __init__(self, n_src: int = 2) -> None:
        super().__init__()
        self.n_src = int(n_src)
        self.register_buffer(
            "perms", torch.tensor(list(permutations(range(n_src))), dtype=torch.long), persistent=False
        )

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        n = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., :n], ref[..., :n]
        matrix = pairwise_si_snr(est, ref)  # (B, est, ref)

        # score every permutation: mean over sources of matrix[b, i, perm[i]]
        scores = []
        for perm in self.perms:
            idx = perm.to(matrix.device)
            picked = matrix[:, torch.arange(self.n_src, device=matrix.device), idx]
            scores.append(picked.mean(dim=-1))
        stacked = torch.stack(scores, dim=1)          # (B, n_perms)
        best, best_idx = stacked.max(dim=1)

        return -best.mean(), {
            "si_snr_db": float(best.mean().detach()),
            "best_permutation": best_idx.detach().cpu().tolist(),
        }

    @torch.no_grad()
    def reorder(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Return ``est`` permuted to line up with ``ref`` (for logging audio)."""
        n = min(est.shape[-1], ref.shape[-1])
        matrix = pairwise_si_snr(est[..., :n], ref[..., :n])
        out = torch.empty_like(est)
        for b in range(est.shape[0]):
            best, best_perm = -1e9, self.perms[0]
            for perm in self.perms:
                score = float(matrix[b, torch.arange(self.n_src), perm].mean())
                if score > best:
                    best, best_perm = score, perm
            out[b] = est[b, best_perm]
        return out


class MixtureConsistencyLoss(nn.Module):
    """Optional regulariser: the sources should add back up to the mixture.

    Cheap, and it noticeably reduces the "energy leaks into nowhere" failure
    mode on real (non-synthetic) recordings.
    """

    def forward(self, est: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        n = min(est.shape[-1], mixture.shape[-1])
        residual = mixture[..., :n] - est[..., :n].sum(dim=1)
        return (residual.pow(2).mean(dim=-1) / (mixture[..., :n].pow(2).mean(dim=-1) + EPS)).mean()
