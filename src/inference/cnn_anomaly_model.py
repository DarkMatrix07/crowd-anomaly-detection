"""CNN Autoencoder-based anomaly model adapter.

Wraps the trained Conv3DAutoencoder so it satisfies the same
``anomaly_model_fn: (np.ndarray) -> float`` interface used by
RollingInferencePipeline — a drop-in replacement for the RF model.

Usage
-----
    from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn

    anomaly_fn = load_cnn_anomaly_model_fn()
    pipeline = RollingInferencePipeline(
        camera_id="cam_01",
        clip_length=30,
        clip_stride=10,
        anomaly_model_fn=anomaly_fn,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.autoencoder import Conv3DAutoencoder

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "autoencoder.pt"

# Preprocessing constants (must match training)
_CLIP_LEN = 16
_FRAME_SIZE = (128, 128)   # (width, height) — must match training

# Reconstruction error normalization.
# Set to ~3x the best validation MSE from training so that normal frames
# score ~0.3 and anomalous frames score higher.  Update after retraining.
_MSE_CAP = 0.005


def _clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
    """Convert raw clip (N, H, W, 3) BGR uint8 to model input tensor (1, 3, T, 64, 64)."""
    n = len(clip)
    # Sub-sample to _CLIP_LEN frames evenly
    indices = np.linspace(0, n - 1, _CLIP_LEN, dtype=int)
    frames = []
    w, h = _FRAME_SIZE
    for i in indices:
        frame = clip[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)

    arr = np.stack(frames, axis=0)          # (T, H, W, 3)
    arr = arr.transpose(3, 0, 1, 2)         # (3, T, H, W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, T, H, W)


def load_cnn_anomaly_model_fn(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> Callable[[np.ndarray], float]:
    """Load trained autoencoder and return an anomaly scoring callable.

    Parameters
    ----------
    model_path : path to a .pt state-dict file, or None to use the default artifact.
    device     : 'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    A function ``(clip: np.ndarray) -> float`` where:
        - ``clip`` is shape (N, H, W, 3) uint8 BGR frames (N >= 1).
        - return value is anomaly probability in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given (or default) path.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"CNN autoencoder not found at {path}.\n"
            "Run scripts/train_autoencoder.py first."
        )

    _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Conv3DAutoencoder()
    model.load_state_dict(torch.load(path, map_location=_device))
    model.to(_device)
    model.eval()

    def anomaly_model_fn(clip: np.ndarray) -> float:
        with torch.no_grad():
            x = _clip_to_tensor(clip).to(_device)
            recon = model(x)
            mse = float(F.mse_loss(recon, x).item())
        # Normalize to [0, 1] using the expected MSE cap
        score = min(1.0, mse / _MSE_CAP)
        return float(np.clip(score, 0.0, 1.0))

    return anomaly_model_fn
