from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)
    return {
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float] | np.ndarray
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        metrics = compute_binary_metrics(y_true, y_score, threshold=float(thr))
        rows.append({"threshold": float(thr), **metrics})
    return pd.DataFrame(rows)


def extract_event_segments(binary_series: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(binary_series).astype(np.int32)
    events: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(binary):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            events.append((start, idx - 1))
            start = None
    if start is not None:
        events.append((start, len(binary) - 1))
    return events


def _segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    intersection = max(0, right - left + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def event_f1_score(
    true_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
    min_iou: float = 0.3,
) -> float:
    matched_true: set[int] = set()
    matched_pred: set[int] = set()

    for pred_idx, pred_event in enumerate(pred_events):
        best_iou = 0.0
        best_true_idx = None
        for true_idx, true_event in enumerate(true_events):
            if true_idx in matched_true:
                continue
            iou = _segment_iou(true_event, pred_event)
            if iou > best_iou:
                best_iou = iou
                best_true_idx = true_idx
        if best_true_idx is not None and best_iou >= min_iou:
            matched_true.add(best_true_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_true)
    fp = len(pred_events) - len(matched_pred)
    fn = len(true_events) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
