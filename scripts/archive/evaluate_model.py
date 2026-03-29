from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.models.evaluate import (
    compute_binary_metrics,
    event_f1_score,
    extract_event_segments,
    threshold_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument("--ckpt", default="", help="Model checkpoint path")
    parser.add_argument(
        "--out",
        default="docs/reports/baseline-threshold-sweep.csv",
        help="CSV path for threshold sweep results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Placeholder demo predictions until full dataset/model integration.
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.35, 0.7, 0.3, 0.88], dtype=np.float32)

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)
    sweep = threshold_sweep(y_true, y_score, thresholds=np.linspace(0.1, 0.9, 9))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_path, index=False)

    event_score = event_f1_score(
        extract_event_segments(y_true),
        extract_event_segments((y_score >= 0.5).astype(np.int32)),
        min_iou=0.3,
    )

    print(f"checkpoint={args.ckpt or '[not provided]'}")
    print(
        f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
        f"f1={metrics['f1']:.4f} event_f1={event_score:.4f}"
    )
    print(f"threshold_sweep_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
