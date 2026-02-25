from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.evaluate import compute_binary_metrics, event_f1_score, extract_event_segments


def _false_alert_rate_per_10m(y_true: np.ndarray, y_pred: np.ndarray, window_minutes: float) -> float:
    negatives = y_true == 0
    false_positives = int(np.sum((y_pred == 1) & negatives))
    if window_minutes <= 0:
        raise ValueError("window_minutes must be > 0")
    return float(false_positives / window_minutes * 10.0)


def run_ablation_suite(
    y_true: np.ndarray,
    variant_scores: dict[str, np.ndarray],
    threshold: float = 0.5,
    window_minutes: float = 1.0,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(np.int32)
    rows: list[dict[str, float | str]] = []

    for variant_name, scores in variant_scores.items():
        y_score = np.asarray(scores).astype(np.float32)
        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
        y_pred = (y_score >= threshold).astype(np.int32)
        evt = event_f1_score(
            extract_event_segments(y_true),
            extract_event_segments(y_pred),
            min_iou=0.3,
        )
        far = _false_alert_rate_per_10m(y_true=y_true, y_pred=y_pred, window_minutes=window_minutes)
        rows.append(
            {
                "variant": variant_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "event_f1": evt,
                "false_alert_rate_per_10m": far,
            }
        )

    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def select_best_tradeoff(results: pd.DataFrame, false_alert_weight: float = 0.1) -> dict[str, float | str]:
    if results.empty:
        raise ValueError("results cannot be empty")
    scored = results.copy()
    scored["tradeoff_score"] = (
        0.7 * scored["f1"] + 0.3 * scored["event_f1"] - false_alert_weight * scored["false_alert_rate_per_10m"]
    )
    best = scored.sort_values("tradeoff_score", ascending=False).iloc[0]
    return best.to_dict()
