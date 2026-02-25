import numpy as np

from src.risk.calibrate import calibrate_thresholds
from src.risk.score import compute_risk_score, smooth_scores


def test_compute_risk_score_weighted_sum() -> None:
    score = compute_risk_score(
        anomaly_score=0.8,
        flow_instability=0.6,
        density_pressure=0.4,
        trend_acceleration=0.2,
        weights={"w1": 0.4, "w2": 0.3, "w3": 0.2, "w4": 0.1},
    )

    assert np.isclose(score, 0.6)


def test_smooth_scores_reduces_spike() -> None:
    series = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32)

    smoothed = smooth_scores(series, alpha=0.4)

    assert smoothed[2] < series[2]
    assert smoothed.shape == series.shape


def test_calibrate_thresholds_returns_ordered_levels() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    scores = np.array([0.1, 0.2, 0.3, 0.65, 0.8, 0.95], dtype=np.float32)

    thresholds = calibrate_thresholds(y_true, scores)

    assert thresholds["low"] < thresholds["medium"] < thresholds["high"]
