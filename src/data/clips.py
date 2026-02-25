from __future__ import annotations

import numpy as np


def generate_clip_windows(
    frames: np.ndarray, clip_length_frames: int, clip_stride_frames: int
) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError("frames must have shape (T, H, W, C)")
    if clip_length_frames <= 0:
        raise ValueError("clip_length_frames must be > 0")
    if clip_stride_frames <= 0:
        raise ValueError("clip_stride_frames must be > 0")

    total_frames = frames.shape[0]
    if total_frames < clip_length_frames:
        return np.empty(
            (0, clip_length_frames, frames.shape[1], frames.shape[2], frames.shape[3]),
            dtype=frames.dtype,
        )

    clips: list[np.ndarray] = []
    for start in range(0, total_frames - clip_length_frames + 1, clip_stride_frames):
        end = start + clip_length_frames
        clips.append(frames[start:end])

    return np.stack(clips, axis=0)
