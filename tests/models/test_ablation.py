import numpy as np

from src.models.ablation import run_ablation_suite, select_best_tradeoff


def test_run_ablation_suite_returns_expected_columns() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    variants = {
        "baseline": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.2, 0.92], dtype=np.float32),
        "no_flow": np.array([0.2, 0.3, 0.7, 0.8, 0.35, 0.7, 0.25, 0.81], dtype=np.float32),
    }

    frame = run_ablation_suite(y_true=y_true, variant_scores=variants, threshold=0.5, window_minutes=1.0)

    required_columns = {
        "variant",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "event_f1",
        "false_alert_rate_per_10m",
    }
    assert required_columns.issubset(frame.columns)
    assert len(frame) == 2


def test_select_best_tradeoff_prefers_higher_score() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    variants = {
        "baseline": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.2, 0.92], dtype=np.float32),
        "no_density": np.array([0.2, 0.3, 0.7, 0.8, 0.45, 0.7, 0.35, 0.75], dtype=np.float32),
    }
    frame = run_ablation_suite(y_true=y_true, variant_scores=variants, threshold=0.5, window_minutes=1.0)

    best = select_best_tradeoff(frame, false_alert_weight=0.05)

    assert best["variant"] == "baseline"
