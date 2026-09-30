"""Neural architectures for the *locally trained* denoiser.

This module imports torch at the top level on purpose -- it is only ever
imported lazily (from the training script, or from
``methods/local_unet.py:load()``), so the web app still starts without torch.

Model: **SpectralUNet**
----------------------
A compact masking U-Net that operates on the log-magnitude STFT and predicts a
real-valued mask in [0, 1] (optionally a complex ratio mask).  Roughly 1.6 M
parameters at the default width -- small enough to train from scratch on a
laptop CPU overnight, big enough to beat every DSP method on its training
domain.

    input  : |STFT| (B, 1, F, T)   F = n_fft/2, T = frames
    output : mask   (B, 1, F, T)   applied to the noisy complex spectrogram
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  building blocks
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: Tuple[int, int] = (2, 1)) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.body = ConvBlock(c_out + c_skip, c_out, stride=(1, 1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.body(torch.cat([x, skip], dim=1))


# --------------------------------------------------------------------------- #
#  the model
# --------------------------------------------------------------------------- #
class SpectralUNet(nn.Module):
    """Mask-predicting U-Net with a bottleneck GRU for temporal context."""

    def __init__(
        self,
        n_fft: int = 512,
        hop: int = 128,
        base_channels: int = 32,
        depth: int = 4,
        use_gru: bool = True,
        mask_type: str = "sigmoid",  # "sigmoid" | "relu" (unbounded) | "crm"
    ) -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.mask_type = mask_type
        self.depth = depth

        chans = [base_channels * (2**i) for i in range(depth)]
        self.stem = ConvBlock(1, chans[0], stride=(1, 1))
        self.downs = nn.ModuleList(
            [ConvBlock(chans[i], chans[i + 1], stride=(2, 1)) for i in range(depth - 1)]
        )

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(chans[-1], chans[-1] // 2, num_layers=1,
                              batch_first=True, bidirectional=True)

        self.ups = nn.ModuleList(
            [UpBlock(chans[i + 1], chans[i], chans[i]) for i in reversed(range(depth - 1))]
        )
        out_ch = 2 if mask_type == "crm" else 1
        self.head = nn.Conv2d(chans[0], out_ch, kernel_size=1)

    # -- spectral helpers -------------------------------------------------- #
    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=wav.device)
        return torch.stft(wav, self.n_fft, self.hop, window=window,
                          return_complex=True, center=True, pad_mode="reflect")

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=spec.device)
        return torch.istft(spec, self.n_fft, self.hop, window=window, center=True, length=length)

    # -- forward ----------------------------------------------------------- #
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) noisy waveform -> (B, T) enhanced waveform."""
        length = wav.shape[-1]
        spec = self.stft(wav)                       # (B, F, T) complex
        mag = spec.abs().unsqueeze(1)               # (B, 1, F, T)
        feat = torch.log1p(mag)

        # drop the Nyquist bin so F is a power of two and the strides line up
        feat = feat[:, :, : self.n_fft // 2, :]

        x = self.stem(feat)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        if self.use_gru:
            b, c, f, t = x.shape
            seq = x.mean(dim=2).transpose(1, 2)      # (B, T, C)
            seq, _ = self.gru(seq)
            x = x + seq.transpose(1, 2).unsqueeze(2).expand(b, c, f, t)

        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 2)])

        mask = self.head(x)
        if self.mask_type == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_type == "relu":
            mask = F.softplus(mask)

        if self.mask_type == "crm":
            mask = torch.tanh(mask)
            real = spec.real[:, : self.n_fft // 2, :] * mask[:, 0] - spec.imag[:, : self.n_fft // 2, :] * mask[:, 1]
            imag = spec.real[:, : self.n_fft // 2, :] * mask[:, 1] + spec.imag[:, : self.n_fft // 2, :] * mask[:, 0]
            enhanced = torch.complex(real, imag)
        else:
            enhanced = spec[:, : self.n_fft // 2, :] * mask[:, 0]

        # put the Nyquist bin back (attenuated -- it is almost never speech)
        nyquist = spec[:, self.n_fft // 2:, :] * 0.1
        enhanced = torch.cat([enhanced, nyquist], dim=1)
        return self.istft(enhanced, length)

    # -- convenience ------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {
            "arch": "SpectralUNet",
            "n_fft": self.n_fft,
            "hop": self.hop,
            "depth": self.depth,
            "use_gru": self.use_gru,
            "mask_type": self.mask_type,
        }


def build_model(cfg: dict) -> SpectralUNet:
    return SpectralUNet(
        n_fft=int(cfg.get("n_fft", 512)),
        hop=int(cfg.get("hop", 128)),
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 4)),
        use_gru=bool(cfg.get("use_gru", True)),
        mask_type=str(cfg.get("mask_type", "sigmoid")),
    )
