from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {"w1": 0.4, "w2": 0.25, "w3": 0.2, "w4": 0.15}


def compute_risk_score(
    anomaly_score: float,
    flow_instability: float,
    density_pressure: float,
    trend_acceleration: float,
    weights: dict[str, float] | None = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS
    score = (
        w["w1"] * anomaly_score
        + w["w2"] * flow_instability
        + w["w3"] * density_pressure
        + w["w4"] * trend_acceleration
    )
    return float(np.clip(score, 0.0, 1.0))


def smooth_scores(scores: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return values
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    out = np.zeros_like(values)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out
