from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import PreprocessConfig, preprocess_video


def _write_tiny_video(path: Path, frames: int = 8, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (80, 60),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create test video at {path}")
    for idx in range(frames):
        frame = np.full((60, 80, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_preprocess_resizes_and_normalizes_without_augment(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_tiny_video(video_path)
    config = PreprocessConfig(
        target_fps=10,
        frame_size=(32, 24),
        clip_length_frames=4,
        clip_stride_frames=2,
        num_workers=0,
        prefetch_factor=2,
        augment_enabled=False,
        horizontal_flip_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        speed_min=1.0,
        speed_max=1.0,
    )

    processed = preprocess_video(video_path, config, seed=7)

    assert processed.shape[1:] == (24, 32, 3)
    assert processed.dtype == np.float32
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0


def test_preprocess_no_augment_mode_is_deterministic(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_tiny_video(video_path)
    config = PreprocessConfig(
        target_fps=10,
        frame_size=(32, 24),
        clip_length_frames=4,
        clip_stride_frames=2,
        num_workers=0,
        prefetch_factor=2,
        augment_enabled=False,
        horizontal_flip_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        speed_min=1.0,
        speed_max=1.0,
    )

    a = preprocess_video(video_path, config, seed=11)
    b = preprocess_video(video_path, config, seed=111)

    np.testing.assert_array_equal(a, b)
