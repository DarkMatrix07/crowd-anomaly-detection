from __future__ import annotations

import numpy as np


def calibrate_thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    scores = np.asarray(scores).astype(np.float32)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")

    negatives = scores[y_true == 0]
    positives = scores[y_true == 1]

    if negatives.size == 0 or positives.size == 0:
        # Fallback defaults when calibration labels are incomplete.
        return {"low": 0.3, "medium": 0.6, "high": 0.85}

    low = float(np.quantile(negatives, 0.90))
    medium = float(np.quantile(scores, 0.60))
    high = float(np.quantile(positives, 0.70))

    ordered = sorted([low, medium, high])
    return {"low": ordered[0], "medium": ordered[1], "high": ordered[2]}
