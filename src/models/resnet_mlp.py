from __future__ import annotations

import torch
from torch import nn


class WindowMLP(nn.Module):
    """MLP classifier on top of temporal-window ResNet18 features.

    Input : (B, in_dim)  — aggregated ResNet18 frame features over a window
    Output: (B, 1)       — anomaly logit (apply sigmoid for probability)
    """

    def __init__(self, in_dim: int = 2048, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
