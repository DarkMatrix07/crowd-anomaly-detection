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
    """Extract 10-dimensional feature vector from a single frame.

    Features:
        0  mean_intensity      – average pixel brightness
        1  std_intensity       – brightness spread (crowd density proxy)
        2  lap_var             – Laplacian variance (edge sharpness / scene activity)
        3  occupancy           – fraction of non-background pixels
        4  flow_mean           – mean optical-flow magnitude
        5  flow_var            – variance of flow magnitudes
        6  flow_max            – peak flow magnitude (spike detector)
        7  directional_entropy – Shannon entropy of flow angle histogram (chaos indicator)
        8  divergence_proxy    – mean |div(flow)| (crowd expansion / panic indicator)
        9  temporal_contrast   – absolute mean-intensity change from previous frame
    """
    if frame is None:
        raise ValueError("frame cannot be None")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    occupancy = float(np.mean(gray > 40))

    flow_mean = 0.0
    flow_var = 0.0
    flow_max = 0.0
    directional_entropy = 0.0
    divergence_proxy = 0.0
    temporal_contrast = 0.0

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        temporal_contrast = float(abs(mean_intensity - float(np.mean(prev_gray))))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u ** 2 + v ** 2)
        flow_mean = float(np.mean(mag))
        flow_var = float(np.var(mag))
        flow_max = float(np.max(mag))

        # Directional entropy: histogram of flow angles → Shannon entropy
        angles = np.arctan2(v, u)  # [-π, π]
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
        hist = hist.astype(np.float64) + 1e-9  # avoid log(0)
        hist /= hist.sum()
        directional_entropy = float(-np.sum(hist * np.log2(hist)))

        # Divergence proxy: ∂u/∂x + ∂v/∂y (finite differences)
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        div = np.abs(du_dx + dv_dy)
        divergence_proxy = float(np.mean(div))

    return np.array(
        [
            mean_intensity,
            std_intensity,
            lap_var,
            occupancy,
            flow_mean,
            flow_var,
            flow_max,
            directional_entropy,
            divergence_proxy,
            temporal_contrast,
        ],
        dtype=np.float32,
    )


def build_labeled_frame_dataset(
    frames_root: Path,
    masks_root: Path,
    frame_stride: int = 3,
    max_frames_per_clip: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load labeled frames from pre-extracted testing frames + mask files.

    Returns
    -------
    x : (N, 10) float32 feature matrix
    y : (N,) int32 label array  (0=normal, 1=anomaly)
    metadata : list of dicts with clip_id, frame_name, frame_index
    """
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
                previous = None  # reset flow across stride gaps
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
                    "source": "testing",
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


def build_normal_dataset_from_videos(
    videos_root: Path,
    frame_stride: int = 5,
    max_frames_per_video: int | None = 300,
    resize: tuple[int, int] | None = (320, 240),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Extract normal-class (label=0) features from raw training video files.

    Parameters
    ----------
    videos_root : directory containing .avi (or .mp4) training video files
    frame_stride : use every nth decoded frame
    max_frames_per_video : cap frames sampled per video (None = unlimited)
    resize : (width, height) to resize frames before feature extraction, or None

    Returns
    -------
    x : (N, 10) float32 feature matrix
    y : (N,) int32 array of zeros (all normal)
    metadata : list of dicts with clip_id, frame_index, source='training'
    """
    if not videos_root.exists():
        raise FileNotFoundError(f"videos_root not found: {videos_root}")

    video_paths = sorted(
        p for p in videos_root.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mov"}
    )
    if not video_paths:
        raise RuntimeError(f"No video files found in {videos_root}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        previous: np.ndarray | None = None
        frame_idx = 0
        used = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                feat = compute_frame_features(frame, prev_frame=previous)
                previous = frame
                features.append(feat)
                labels.append(0)
                metadata.append(
                    {
                        "clip_id": video_path.stem,
                        "frame_name": f"{frame_idx:06d}.jpg",
                        "frame_index": frame_idx,
                        "source": "training",
                    }
                )
                used += 1
                if max_frames_per_video is not None and used >= max_frames_per_video:
                    break
            frame_idx += 1

        cap.release()

    if not features:
        raise RuntimeError("No frames extracted from training videos.")
    x = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y, metadata
