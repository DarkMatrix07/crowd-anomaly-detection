from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def iter_video_frames(source: str, max_frames: int | None = None) -> Iterator[tuple[np.ndarray, float]]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open stream source: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 10.0

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            yield frame, timestamp
            index += 1
            if max_frames is not None and index >= max_frames:
                break
    finally:
        capture.release()
