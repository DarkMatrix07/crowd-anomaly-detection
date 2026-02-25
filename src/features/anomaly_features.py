from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _sorted_frame_paths(clip_dir: Path) -> list[Path]:
    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem
        try:
            return int(stem), stem
        except ValueError:
            return 0, stem

    return sorted((p for p in clip_dir.glob("*.jpg") if p.is_file()), key=_key)


def compute_frame_features(frame: np.ndarray, prev_frame: np.ndarray | None = None) -> np.ndarray:
    if frame is None:
        raise ValueError("frame cannot be None")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    occupancy = float(np.mean(gray > 40))

    flow_mean = 0.0
    flow_var = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=9,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag = np.linalg.norm(flow, axis=-1)
        flow_mean = float(np.mean(mag))
        flow_var = float(np.var(mag))

    return np.array(
        [mean_intensity, std_intensity, lap_var, occupancy, flow_mean, flow_var],
        dtype=np.float32,
    )


def build_labeled_frame_dataset(
    frames_root: Path,
    masks_root: Path,
    frame_stride: int = 3,
    max_frames_per_clip: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0")
    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root not found: {frames_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    clip_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    for clip_dir in clip_dirs:
        mask_path = masks_root / f"{clip_dir.name}.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(np.uint8).reshape(-1)
        frame_paths = _sorted_frame_paths(clip_dir)
        if not frame_paths:
            continue

        previous = None
        used = 0
        for idx, frame_path in enumerate(frame_paths):
            if idx >= len(mask):
                break
            if idx % frame_stride != 0:
                continue
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            feat = compute_frame_features(frame, prev_frame=previous)
            previous = frame
            features.append(feat)
            labels.append(int(mask[idx]))
            metadata.append(
                {
                    "clip_id": clip_dir.name,
                    "frame_name": frame_path.name,
                    "frame_index": idx,
                }
            )
            used += 1
            if max_frames_per_clip is not None and used >= max_frames_per_clip:
                break

    if not features:
        raise RuntimeError("No labeled frames were loaded from provided roots.")
    x = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y, metadata
