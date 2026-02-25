from pathlib import Path

import numpy as np

from src.features.optical_flow import compute_optical_flow, save_flow_visualization


def _moving_square_frames(frame_count: int = 3) -> np.ndarray:
    frames = []
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        left = 8 + idx * 3
        frame[16:32, left : left + 12, :] = 255
        frames.append(frame)
    return np.stack(frames, axis=0)


def test_compute_optical_flow_shape_and_direction() -> None:
    frames = _moving_square_frames()

    flow = compute_optical_flow(frames, method="farneback")

    assert flow.shape == (2, 48, 64, 2)
    # Motion should be mostly toward +x for rightward movement.
    assert float(flow[..., 0].mean()) > 0.0


def test_save_flow_visualization_writes_image(tmp_path: Path) -> None:
    frames = _moving_square_frames()
    flow = compute_optical_flow(frames, method="farneback")
    output = tmp_path / "flow_vis.png"

    save_flow_visualization(flow[0], output)

    assert output.exists()
    assert output.stat().st_size > 0
