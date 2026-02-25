from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np


def _as_uint8_rgb_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    if frames.dtype == np.uint8:
        return frames
    clipped = np.clip(frames, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def _farneback_flow(frames: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb_frames(frames)
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    flows: list[np.ndarray] = []
    for prev, nxt in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow.astype(np.float32))
    if not flows:
        h, w = frames.shape[1], frames.shape[2]
        return np.zeros((0, h, w, 2), dtype=np.float32)
    return np.stack(flows, axis=0)


def compute_optical_flow(
    frames: np.ndarray, method: str = "raft", raft_model: object | None = None
) -> np.ndarray:
    normalized_method = method.lower()

    # RAFT is the preferred path for GPU training/inference. We keep
    # Farneback as an always-available fallback in this scaffold.
    if normalized_method == "raft":
        if raft_model is None:
            warnings.warn(
                "RAFT model not provided; falling back to Farneback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _farneback_flow(frames)
        warnings.warn(
            "RAFT execution is not yet wired; using Farneback fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _farneback_flow(frames)

    if normalized_method == "farneback":
        return _farneback_flow(frames)

    raise ValueError("method must be 'raft' or 'farneback'")


def save_flow_visualization(flow_frame: np.ndarray, output_path: Path) -> Path:
    if flow_frame.ndim != 3 or flow_frame.shape[-1] != 2:
        raise ValueError("flow_frame must have shape (H, W, 2)")

    mag, ang = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1], angleInDegrees=False)
    hsv = np.zeros((flow_frame.shape[0], flow_frame.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(ang * 90 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), bgr)
    if not ok:
        raise IOError(f"Failed to write flow visualization to {output_path}")
    return output_path
