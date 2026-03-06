from __future__ import annotations

import torch
from torch import nn


class Conv3DAutoencoder(nn.Module):
    """Conv3D autoencoder for video anomaly detection.

    Trained on normal-only clips. High reconstruction error => anomaly.
    Input shape: (B, C=3, T=16, H=64, W=64)
    Requires input normalized to [0, 1] (pixel values / 255.0).
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Encoder: downsample T by 4, H/W by 8
        # (B, 3, 16, 64, 64) -> (B, c*4, 4, 8, 8)
        self.encoder = nn.Sequential(
            # (B, 3, 16, 64, 64) -> (B, c, 16, 32, 32)
            nn.Conv3d(3, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            # -> (B, c*2, 8, 16, 16)
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # -> (B, c*4, 4, 8, 8)
            nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        # Decoder: upsample back to original shape
        # (B, c*4, 4, 8, 8) -> (B, 3, 16, 64, 64)
        self.decoder = nn.Sequential(
            # (B, c*4, 4, 8, 8) -> (B, c*2, 8, 16, 16)
            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(),
            # -> (B, c, 16, 32, 32)
            nn.ConvTranspose3d(c * 2, c, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(c),
            nn.ReLU(),
            # -> (B, 3, 16, 64, 64)
            nn.ConvTranspose3d(c, 3, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the compressed latent representation."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode; output has same shape as input."""
        return self.decoder(self.encode(x))
