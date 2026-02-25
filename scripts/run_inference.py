from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import RollingInferencePipeline
from src.inference.stream import iter_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling inference on stream or video")
    parser.add_argument("--source", default="", help="RTSP URL or local video file")
    parser.add_argument("--config", default="configs/infer.yaml", help="Inference config path")
    parser.add_argument("--camera-id", default="cam-1", help="Camera identifier")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    return parser.parse_args()


def _default_anomaly_model(clip: np.ndarray) -> float:
    # Placeholder anomaly score until model checkpoint inference is integrated.
    return float(np.clip(np.mean(clip) / 255.0, 0.0, 1.0))


def _synthetic_stream(max_frames: int) -> list[tuple[np.ndarray, float]]:
    rng = np.random.default_rng(123)
    frames = []
    for idx in range(max_frames):
        frame = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        frames.append((frame, idx * 0.1))
    return frames


def main() -> int:
    args = parse_args()
    pipeline = RollingInferencePipeline(
        camera_id=args.camera_id,
        clip_length=8,
        clip_stride=4,
        anomaly_model_fn=_default_anomaly_model,
    )

    if args.source:
        frame_iter = iter_video_frames(args.source, max_frames=args.max_frames)
    else:
        frame_iter = _synthetic_stream(args.max_frames)

    event_count = 0
    for frame, timestamp in frame_iter:
        for event in pipeline.process_frame(frame=frame, timestamp=timestamp):
            print(json.dumps(event))
            event_count += 1
    print(f"events_emitted={event_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
