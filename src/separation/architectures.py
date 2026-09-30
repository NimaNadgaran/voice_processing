"""Neural architecture for the *locally trained* separator: Conv-TasNet.

Imported lazily (training script / ``methods/local_convtasnet.py:load()``), so
the app still runs without torch.

Conv-TasNet (Luo & Mesgarani, 2019) in three parts:

``Encoder``   1-D convolution with stride L/2 turns the waveform into a
              non-negative "spectrogram-like" latent of N channels.
``Separator`` a stack of R repeats x X dilated depthwise-separable conv blocks
              (a temporal convolutional network) predicts one mask per speaker.
``Decoder``   1-D transposed convolution maps each masked latent back to audio.

Trained with utterance-level permutation-invariant SI-SNR loss (see
``training/losses.py``) -- the loss tries every assignment of outputs to
references and keeps the best, which is what makes the model source-agnostic.

Default size below is deliberately *small* (~3.5 M params) so it is trainable on
a laptop; the paper configuration is roughly 5 M.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  normalisation
# --------------------------------------------------------------------------- #
class GlobalLayerNorm(nn.Module):
    """gLN: normalise over both channel and time (non-causal)."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class CumulativeLayerNorm(nn.Module):
    """cLN: causal variant -- statistics only from the past."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        step = torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, 1, t) * c
        csum = x.sum(dim=1, keepdim=True).cumsum(dim=2)
        csum_sq = (x**2).sum(dim=1, keepdim=True).cumsum(dim=2)
        mean = csum / step
        var = csum_sq / step - mean**2
        return self.gamma * (x - mean) / torch.sqrt(var.clamp(min=self.eps)) + self.beta


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "gLN":
        return GlobalLayerNorm(channels)
    if kind == "cLN":
        return CumulativeLayerNorm(channels)
    return nn.BatchNorm1d(channels)


# --------------------------------------------------------------------------- #
#  TCN block
# --------------------------------------------------------------------------- #
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel: int, dilation: int, norm: str = "gLN", causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.padding = (kernel - 1) * dilation if causal else (kernel - 1) * dilation // 2

        self.conv1x1 = nn.Conv1d(in_ch, hidden, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = make_norm(norm, hidden)
        self.dconv = nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=self.padding, groups=hidden)
        self.prelu2 = nn.PReLU()
        self.norm2 = make_norm(norm, hidden)
        self.res_out = nn.Conv1d(hidden, in_ch, 1)
        self.skip_out = nn.Conv1d(hidden, in_ch, 1)

    def forward(self, x: torch.Tensor):
        y = self.norm1(self.prelu1(self.conv1x1(x)))
        y = self.dconv(y)
        if self.causal and self.padding:
            y = y[..., : -self.padding]
        y = self.norm2(self.prelu2(y))
        return x + self.res_out(y), self.skip_out(y)


# --------------------------------------------------------------------------- #
#  the model
# --------------------------------------------------------------------------- #
class ConvTasNet(nn.Module):
    def __init__(
        self,
        n_src: int = 2,
        enc_channels: int = 256,   # N
        enc_kernel: int = 16,      # L  (2 ms at 8 kHz)
        bottleneck: int = 128,     # B
        hidden: int = 256,         # H
        kernel: int = 3,           # P
        n_blocks: int = 7,         # X
        n_repeats: int = 2,        # R
        norm: str = "gLN",
        causal: bool = False,
        mask_act: str = "relu",
        sample_rate: int = 8000,
    ) -> None:
        super().__init__()
        self.n_src = n_src
        self.enc_kernel = enc_kernel
        self.stride = enc_kernel // 2
        self.sample_rate = sample_rate
        self.mask_act = mask_act
        self._cfg = dict(
            n_src=n_src, enc_channels=enc_channels, enc_kernel=enc_kernel, bottleneck=bottleneck,
            hidden=hidden, kernel=kernel, n_blocks=n_blocks, n_repeats=n_repeats, norm=norm,
            causal=causal, mask_act=mask_act, sample_rate=sample_rate,
        )

        self.encoder = nn.Conv1d(1, enc_channels, enc_kernel, stride=self.stride, bias=False)
        self.ln = make_norm("cLN" if causal else "gLN", enc_channels)
        self.bottleneck_conv = nn.Conv1d(enc_channels, bottleneck, 1)

        blocks = []
        for _ in range(n_repeats):
            for b in range(n_blocks):
                blocks.append(TCNBlock(bottleneck, hidden, kernel, 2**b, norm, causal))
        self.tcn = nn.ModuleList(blocks)

        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(bottleneck, n_src * enc_channels, 1)
        self.decoder = nn.ConvTranspose1d(enc_channels, 1, enc_kernel, stride=self.stride, bias=False)

    # -- forward ----------------------------------------------------------- #
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) mixture -> (B, n_src, T) estimated sources."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B, 1, T)
        length = wav.shape[-1]

        pad = self._pad_amount(length)
        if pad:
            wav = F.pad(wav, (0, pad))

        w = F.relu(self.encoder(wav))                    # (B, N, K)
        y = self.bottleneck_conv(self.ln(w))             # (B, B, K)

        skip_sum = torch.zeros_like(y)
        for block in self.tcn:
            y, skip = block(y)
            skip_sum = skip_sum + skip

        masks = self.mask_conv(self.mask_prelu(skip_sum))  # (B, n_src*N, K)
        b, _, k = masks.shape
        masks = masks.view(b, self.n_src, -1, k)
        masks = torch.sigmoid(masks) if self.mask_act == "sigmoid" else F.relu(masks)

        masked = w.unsqueeze(1) * masks                   # (B, n_src, N, K)
        out = self.decoder(masked.view(b * self.n_src, -1, k))
        out = out.view(b, self.n_src, -1)
        return out[..., :length]

    def _pad_amount(self, length: int) -> int:
        stride = self.stride
        rest = (length - self.enc_kernel) % stride
        return 0 if rest == 0 else stride - rest

    # -- convenience ------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict:
        cfg = dict(self._cfg)
        cfg["arch"] = "ConvTasNet"
        return cfg


def build_model(cfg: Dict) -> ConvTasNet:
    return ConvTasNet(
        n_src=int(cfg.get("n_src", 2)),
        enc_channels=int(cfg.get("enc_channels", 256)),
        enc_kernel=int(cfg.get("enc_kernel", 16)),
        bottleneck=int(cfg.get("bottleneck", 128)),
        hidden=int(cfg.get("hidden", 256)),
        kernel=int(cfg.get("kernel", 3)),
        n_blocks=int(cfg.get("n_blocks", 7)),
        n_repeats=int(cfg.get("n_repeats", 2)),
        norm=str(cfg.get("norm", "gLN")),
        causal=bool(cfg.get("causal", False)),
        mask_act=str(cfg.get("mask_act", "relu")),
        sample_rate=int(cfg.get("sample_rate", 8000)),
    )
