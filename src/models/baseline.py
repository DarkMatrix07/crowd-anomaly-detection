from __future__ import annotations

import torch
from torch import nn


class Conv3DBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
