import numpy as np

from src.models.evaluate import (
    compute_binary_metrics,
    event_f1_score,
    extract_event_segments,
    threshold_sweep,
)


def test_compute_binary_metrics_for_separable_scores() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.05, 0.2, 0.8, 0.95], dtype=np.float32)

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics["roc_auc"] >= 0.99
    assert metrics["pr_auc"] >= 0.99
    assert metrics["f1"] >= 0.99


def test_threshold_sweep_returns_row_per_threshold() -> None:
    y_true = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.2, 0.9, 0.4, 0.7, 0.6], dtype=np.float32)

    sweep = threshold_sweep(y_true, y_score, thresholds=[0.3, 0.5, 0.7])

    assert len(sweep) == 3
    assert set(sweep.columns) >= {"threshold", "f1", "precision", "recall"}


def test_event_f1_score_detects_overlap() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int32)

    true_events = extract_event_segments(y_true)
    pred_events = extract_event_segments(y_pred)
    score = event_f1_score(true_events, pred_events, min_iou=0.3)

    assert score > 0.5
