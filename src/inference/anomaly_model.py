"""Windowed anomaly model loader for the inference pipeline.

Wraps the trained temporal-window RandomForest so it can be plugged into
RollingInferencePipeline as the ``anomaly_model_fn`` callable.

Usage
-----
    from src.inference.anomaly_model import load_anomaly_model_fn

    anomaly_fn = load_anomaly_model_fn()          # loads default artifact
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
import joblib
import numpy as np

from src.features.anomaly_features import compute_frame_features

# Default model path relative to project root
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "shanghaitech_windowed_rf.joblib"

# Window feature dimensionality: mean + std + max + delta of 10 frame features
_FRAME_DIM = 10
_WINDOW_DIM = _FRAME_DIM * 4  # 40


def _clip_to_window_features(clip: np.ndarray) -> np.ndarray:
    """Convert a clip (N, H, W, 3) uint8 array to a single 40-d window feature vector.

    Steps
    -----
    1. Resize each frame to 320×240 (matches training resolution).
    2. Extract 10-d frame features with optical flow between consecutive frames.
    3. Aggregate over all N frames: mean, std, max, delta (last-third − first-third).

    Returns
    -------
    np.ndarray of shape (1, 40) float32 — ready for model.predict_proba().
    """
    n_frames = len(clip)
    frame_feats: list[np.ndarray] = []
    previous: np.ndarray | None = None

    for i in range(n_frames):
        frame = clip[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        feat = compute_frame_features(frame, prev_frame=previous)
        previous = frame
        frame_feats.append(feat)

    if not frame_feats:
        return np.zeros((1, _WINDOW_DIM), dtype=np.float32)

    arr = np.vstack(frame_feats).astype(np.float32)   # (N, 10)
    third = max(1, n_frames // 3)

    f_mean  = arr.mean(axis=0)
    f_std   = arr.std(axis=0)
    f_max   = arr.max(axis=0)
    f_delta = arr[-third:].mean(axis=0) - arr[:third].mean(axis=0)

    return np.concatenate([f_mean, f_std, f_max, f_delta]).reshape(1, -1)


def load_anomaly_model_fn(
    model_path: str | Path | None = None,
) -> Callable[[np.ndarray], float]:
    """Load the trained windowed model and return an anomaly scoring callable.

    Parameters
    ----------
    model_path : path to a .joblib model file, or None to use the default artifact.

    Returns
    -------
    A function ``(clip: np.ndarray) -> float`` where:
        - ``clip`` is shape (N, H, W, 3) uint8 BGR frames (N ≥ 1).
        - return value is an anomaly probability in [0, 1].
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly model not found at {path}.\n"
            "Run scripts/train_windowed.py first to generate the artifact."
        )
    model = joblib.load(path)

    def anomaly_model_fn(clip: np.ndarray) -> float:
        window_feat = _clip_to_window_features(clip)
        prob = float(model.predict_proba(window_feat)[0, 1])
        return float(np.clip(prob, 0.0, 1.0))

    return anomaly_model_fn
