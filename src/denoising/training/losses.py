"""Losses for the local denoiser.

The combination that actually works in practice (and what the default config
uses) is ``SI-SNR + multi-resolution STFT``:

* **SI-SNR** is scale invariant and optimises the waveform directly, which
  keeps the phase honest;
* **multi-resolution STFT** (Yamamoto et al., 2020) adds a magnitude term at
  three different window sizes.  It is what stops the model from producing the
  "smeared / underwater" artefacts that a pure time-domain loss allows.

Both are cheap. Weighted 1.0 / 0.5 by default.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B,) scale-invariant SNR in dB. Higher is better."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(dim=-1) + EPS) / (noise.pow(2).sum(dim=-1) + EPS))


class SISNRLoss(nn.Module):
    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return -si_snr(est, ref).mean()


class STFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 at one resolution."""

    def __init__(self, n_fft: int = 1024, hop: int = 256, win: int = 1024) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def _mag(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, self.n_fft, self.hop, self.win, window=self.window,
            return_complex=True, center=True, pad_mode="reflect",
        )
        return spec.abs().clamp(min=EPS)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        est_mag, ref_mag = self._mag(est), self._mag(ref)
        convergence = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + EPS)
        log_l1 = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
        return convergence + log_l1


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        hops: Sequence[int] = (128, 256, 512),
        wins: Sequence[int] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.losses = nn.ModuleList(
            [STFTLoss(n, h, w) for n, h, w in zip(n_ffts, hops, wins)]
        )

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return sum(loss(est, ref) for loss in self.losses) / len(self.losses)


class CombinedLoss(nn.Module):
    """``w_sisnr * SI-SNR + w_stft * multi-res STFT`` with per-term reporting."""

    def __init__(self, w_sisnr: float = 1.0, w_stft: float = 0.5) -> None:
        super().__init__()
        self.sisnr = SISNRLoss()
        self.stft = MultiResolutionSTFTLoss()
        self.w_sisnr = float(w_sisnr)
        self.w_stft = float(w_stft)

    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        n = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., :n], ref[..., :n]
        l_sisnr = self.sisnr(est, ref)
        l_stft = self.stft(est, ref) if self.w_stft > 0 else torch.zeros((), device=est.device)
        total = self.w_sisnr * l_sisnr + self.w_stft * l_stft
        return total, {"sisnr_db": float(-l_sisnr.detach()), "stft": float(l_stft.detach())}
