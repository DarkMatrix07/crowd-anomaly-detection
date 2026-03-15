from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Conv3DAutoencoder(nn.Module):
    """Conv3D autoencoder for video anomaly detection (NO skip connections).

    Plain encoder → bottleneck → decoder.  Skip connections are intentionally
    omitted so the bottleneck must compress all information; unseen anomalous
    patterns cannot be reconstructed well, producing a high MSE anomaly signal.

    Input shape : (B, C=3, T=16, H=128, W=128)
    Output shape: same as input
    Requires input normalised to [0, 1].
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _conv_block(3, c)             # (B, c,   16, 128, 128)
        self.pool1 = nn.MaxPool3d((1, 2, 2))      # (B, c,   16,  64,  64)

        self.enc2 = _conv_block(c, c * 2)         # (B, c*2, 16,  64,  64)
        self.pool2 = nn.MaxPool3d((2, 2, 2))      # (B, c*2,  8,  32,  32)

        self.enc3 = _conv_block(c * 2, c * 4)    # (B, c*4,  8,  32,  32)
        self.pool3 = nn.MaxPool3d((2, 2, 2))      # (B, c*4,  4,  16,  16)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            _conv_block(c * 4, c * 8),
            _conv_block(c * 8, c * 8),
        )                                          # (B, c*8,  4,  16,  16)

        # ── Decoder (NO skip connections) ────────────────────────────────────
        self.up3  = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec3 = _conv_block(c * 4, c * 4)    # input = up3 only

        self.up2  = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = _conv_block(c * 2, c * 2)    # input = up2 only

        self.up1  = nn.ConvTranspose3d(c * 2, c, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = _conv_block(c, c)             # input = up1 only

        # ── Output projection ────────────────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv3d(c, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck latent representation."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        return self.bottleneck(self.pool3(s3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                          # (B, c,   16, 128, 128)
        e2 = self.enc2(self.pool1(e1))             # (B, c*2, 16,  64,  64)
        e3 = self.enc3(self.pool2(e2))             # (B, c*4,  8,  32,  32)
        bn = self.bottleneck(self.pool3(e3))       # (B, c*8,  4,  16,  16)

        # Decoder — no skip connections, bottleneck only
        d3 = self.dec3(self.up3(bn))               # (B, c*4,  8,  32,  32)
        d2 = self.dec2(self.up2(d3))               # (B, c*2, 16,  64,  64)
        d1 = self.dec1(self.up1(d2))               # (B, c,   16, 128, 128)

        return self.out_conv(d1)                   # (B, 3, 16, 128, 128)
