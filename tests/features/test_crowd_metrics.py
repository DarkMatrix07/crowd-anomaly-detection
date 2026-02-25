import numpy as np

from src.features.crowd_metrics import (
    build_feature_row,
    compute_density_pressure,
    compute_local_concentration_index,
    compute_motion_statistics,
)
from src.features.optical_flow import compute_optical_flow


def _synthetic_frames(frame_count: int = 4) -> np.ndarray:
    frames = []
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[8:40, 10 + idx : 38 + idx, :] = 160
        frames.append(frame)
    return np.stack(frames, axis=0)


def test_motion_statistics_have_expected_keys() -> None:
    frames = _synthetic_frames()
    flow = compute_optical_flow(frames, method="farneback")

    metrics = compute_motion_statistics(flow)

    assert set(metrics) == {
        "flow_mean_magnitude",
        "flow_variance_magnitude",
        "flow_directional_entropy",
        "flow_divergence_proxy",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_density_metrics_return_finite_values() -> None:
    frames = _synthetic_frames()

    density_pressure = compute_density_pressure(frames)
    concentration = compute_local_concentration_index(frames, grid_size=4)

    assert np.isfinite(density_pressure)
    assert np.isfinite(concentration)
    assert density_pressure > 0.0


def test_build_feature_row_combines_motion_and_density() -> None:
    frames = _synthetic_frames()
    flow = compute_optical_flow(frames, method="farneback")

    row = build_feature_row(frames, flow)

    expected = {
        "flow_mean_magnitude",
        "flow_variance_magnitude",
        "flow_directional_entropy",
        "flow_divergence_proxy",
        "density_pressure",
        "local_concentration_index",
    }
    assert set(row) == expected
    assert all(np.isfinite(v) for v in row.values())
