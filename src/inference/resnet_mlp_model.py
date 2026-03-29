"""ResNet18 + MLP anomaly model adapter.

Drop-in replacement for the RF and CNN autoencoder adapters.
Satisfies the ``anomaly_model_fn: (np.ndarray) -> float`` interface.

Usage
-----
    from src.inference.resnet_mlp_model import load_resnet_mlp_model_fn

    anomaly_fn = load_resnet_mlp_model_fn()
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
from torchvision import models, transforms

from src.models.resnet_mlp import WindowMLP

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "resnet_mlp.pt"

_FRAME_SIZE    = (224, 224)
_FEAT_DIM      = 512
_AGG_DIM       = _FEAT_DIM * 4   # 2048
_WINDOW_SIZE   = 30

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, _FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return _transform(rgb)


def load_resnet_mlp_model_fn(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> Callable[[np.ndarray], float]:
    """Load trained ResNet18+MLP and return an anomaly scoring callable.

    Parameters
    ----------
    model_path : path to .pt state-dict, or None to use default artifact.
    device     : 'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    ``(clip: np.ndarray) -> float`` — clip is (N, H, W, 3) uint8 BGR,
    returns anomaly probability in [0, 1].
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"ResNet+MLP model not found at {path}.\n"
            "Run scripts/train_resnet_mlp.py first."
        )

    _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Frozen backbone
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone = backbone.to(_device).eval()

    # MLP classifier
    mlp = WindowMLP(in_dim=_AGG_DIM)
    mlp.load_state_dict(torch.load(path, map_location=_device))
    mlp = mlp.to(_device).eval()

    def anomaly_model_fn(clip: np.ndarray) -> float:
        n = len(clip)
        with torch.no_grad():
            tensors = torch.stack([_frame_to_tensor(clip[i]) for i in range(n)]).to(_device)
            feats = backbone(tensors).cpu().numpy()   # (N, 512)

        # Use up to _WINDOW_SIZE frames (sub-sample if longer)
        if n > _WINDOW_SIZE:
            idx = np.linspace(0, n - 1, _WINDOW_SIZE, dtype=int)
            feats = feats[idx]

        third = max(1, len(feats) // 3)
        agg = np.concatenate([
            feats.mean(axis=0),
            feats.std(axis=0),
            feats.max(axis=0),
            feats[-third:].mean(axis=0) - feats[:third].mean(axis=0),
        ]).astype(np.float32)   # (2048,)

        x = torch.from_numpy(agg).unsqueeze(0).to(_device)
        with torch.no_grad():
            logit = mlp(x).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        return float(np.clip(prob, 0.0, 1.0))

    return anomaly_model_fn
