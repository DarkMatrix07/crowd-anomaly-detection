from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    target_fps: int
    frame_size: tuple[int, int]
    clip_length_frames: int
    clip_stride_frames: int
    num_workers: int
    prefetch_factor: int
    augment_enabled: bool
    horizontal_flip_prob: float
    brightness_jitter: float
    contrast_jitter: float
    speed_min: float
    speed_max: float


def _read_video(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames, fps


def _resample_frames(
    frames: list[np.ndarray], source_fps: float, target_fps: int
) -> list[np.ndarray]:
    if target_fps <= 0 or source_fps <= 0:
        return frames
    if source_fps <= target_fps:
        return frames
    stride = max(1, int(round(source_fps / target_fps)))
    return frames[::stride]


def _apply_augmentations(
    frames: list[np.ndarray], config: PreprocessConfig, rng: np.random.Generator
) -> list[np.ndarray]:
    if not frames:
        return frames

    augmented = list(frames)

    if config.speed_min > 0 and config.speed_max > 0 and config.speed_max >= config.speed_min:
        speed = float(rng.uniform(config.speed_min, config.speed_max))
        if abs(speed - 1.0) > 1e-3:
            stride = max(1, int(round(speed)))
            augmented = augmented[::stride]
            if not augmented:
                augmented = [frames[0]]

    if rng.random() < config.horizontal_flip_prob:
        augmented = [cv2.flip(frame, 1) for frame in augmented]

    contrast = 1.0 + float(rng.uniform(-config.contrast_jitter, config.contrast_jitter))
    brightness = float(rng.uniform(-config.brightness_jitter, config.brightness_jitter)) * 255.0
    jittered = []
    for frame in augmented:
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        jittered.append(adjusted)
    return jittered


def preprocess_video(
    video_path: Path, config: PreprocessConfig, seed: int | None = None
) -> np.ndarray:
    frames, source_fps = _read_video(video_path)
    frames = _resample_frames(frames, source_fps=source_fps, target_fps=config.target_fps)

    if config.augment_enabled:
        rng = np.random.default_rng(seed)
        frames = _apply_augmentations(frames, config, rng=rng)

    target_width, target_height = config.frame_size
    processed: list[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        processed.append(normalized)

    return np.stack(processed, axis=0)
