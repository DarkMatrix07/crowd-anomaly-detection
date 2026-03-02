from pathlib import Path

import cv2
import numpy as np

from src.features.anomaly_features import (
    build_labeled_frame_dataset,
    compute_frame_features,
)


def _make_moving_frames() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((48, 64, 3), dtype=np.uint8)
    b = np.zeros((48, 64, 3), dtype=np.uint8)
    a[12:28, 10:26, :] = 255
    b[12:28, 14:30, :] = 255
    return a, b


def test_compute_frame_features_length_and_motion_signal() -> None:
    prev_frame, frame = _make_moving_frames()
    feat = compute_frame_features(frame, prev_frame=prev_frame)
    assert feat.shape == (10,)
    assert float(feat[-2]) > 0.0


def test_build_labeled_frame_dataset_reads_frames_and_masks(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    masks_root = tmp_path / "test_frame_mask"
    clip_dir = frames_root / "01_0001"
    clip_dir.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    for idx in range(4):
        img = np.full((32, 32, 3), idx * 50, dtype=np.uint8)
        cv2.imwrite(str(clip_dir / f"{idx:03d}.jpg"), img)

    np.save(masks_root / "01_0001.npy", np.array([0, 0, 1, 1], dtype=np.uint8))

    x, y, meta = build_labeled_frame_dataset(frames_root, masks_root, frame_stride=1)

    assert x.shape[0] == 4
    assert y.tolist() == [0, 0, 1, 1]
    assert len(meta) == 4
