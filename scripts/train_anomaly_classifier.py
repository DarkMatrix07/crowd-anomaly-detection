from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import build_labeled_frame_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShanghaiTech anomaly classifier with 70/30 split")
    parser.add_argument("--dataset-root", default="data/raw/shanghaitech/shanghaitech", help="ShanghaiTech root path")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--frame-stride", type=int, default=3, help="Use every nth frame")
    parser.add_argument("--max-frames-per-clip", type=int, default=180, help="Cap frames loaded per clip")
    parser.add_argument("--n-estimators", type=int, default=400, help="RandomForest tree count")
    parser.add_argument(
        "--model-out",
        default="artifacts/models/shanghaitech_anomaly_rf_70_30.joblib",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/reports/shanghaitech_anomaly_rf_70_30_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--predictions-out",
        default="artifacts/reports/shanghaitech_anomaly_rf_70_30_predictions.csv",
        help="Output predictions CSV path",
    )
    parser.add_argument(
        "--manifest-out",
        default="data/interim/shanghaitech_anomaly_70_30_manifest.csv",
        help="Output split manifest CSV path",
    )
    return parser.parse_args()


def _build_model(n_estimators: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _save_manifest(meta: list[dict[str, object]], train_idx: np.ndarray, test_idx: np.ndarray, path: Path) -> None:
    train_set = set(train_idx.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "clip_id", "frame_name", "frame_index", "split"])
        for idx, row in enumerate(meta):
            split = "train" if idx in train_set else "test"
            writer.writerow([idx, row["clip_id"], row["frame_name"], row["frame_index"], split])


def _save_predictions(
    meta: list[dict[str, object]],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "clip_id", "frame_name", "frame_index", "y_true", "y_pred", "y_prob"])
        for local_i, idx in enumerate(indices):
            row = meta[int(idx)]
            writer.writerow(
                [
                    int(idx),
                    row["clip_id"],
                    row["frame_name"],
                    row["frame_index"],
                    int(y_true[local_i]),
                    int(y_pred[local_i]),
                    float(y_prob[local_i]),
                ]
            )


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    frames_root = dataset_root / "testing" / "frames"
    masks_root = dataset_root / "testing" / "test_frame_mask"

    x, y, meta = build_labeled_frame_dataset(
        frames_root=frames_root,
        masks_root=masks_root,
        frame_stride=args.frame_stride,
        max_frames_per_clip=args.max_frames_per_clip,
    )

    indices = np.arange(len(y))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        indices,
        train_size=args.train_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=y,
    )

    model = _build_model(n_estimators=args.n_estimators, seed=args.seed)
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "dataset_root": str(dataset_root),
        "samples_total": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_ratio": float(len(y_train) / len(y)),
        "test_ratio": float(len(y_test) / len(y)),
        "positive_rate_total": float(np.mean(y)),
        "accuracy": acc,
        "error_rate": float(1.0 - acc),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "model": "RandomForestClassifier",
        "n_estimators": int(args.n_estimators),
        "frame_stride": int(args.frame_stride),
        "max_frames_per_clip": int(args.max_frames_per_clip),
        "seed": int(args.seed),
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _save_manifest(meta, idx_train, idx_test, Path(args.manifest_out))
    _save_predictions(meta, idx_test, y_test, y_pred, y_prob, Path(args.predictions_out))

    print(f"train_samples={len(y_train)} test_samples={len(y_test)}")
    print(
        f"accuracy={acc:.4f} error_rate={1.0-acc:.4f} precision={precision:.4f} "
        f"recall={recall:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f} pr_auc={pr_auc:.4f}"
    )
    print(f"metrics={metrics_out}")
    print(f"model={model_out}")
    print(f"manifest={Path(args.manifest_out)}")
    print(f"predictions={Path(args.predictions_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
