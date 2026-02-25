from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.shanghaitech_count import collect_samples, load_features_and_targets, split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ShanghaiTech crowd-count baseline using 70/30 split")
    parser.add_argument("--dataset-root", default="ShanghaiTech", help="Path to ShanghaiTech dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest tree count")
    parser.add_argument("--image-size", type=int, default=32, help="Square grayscale resize size")
    parser.add_argument("--epochs", type=int, default=1, help="Compatibility arg (not used by RF model)")
    parser.add_argument(
        "--model-out", default="artifacts/models/shanghaitech_rf_70_30.joblib", help="Path to save trained model"
    )
    parser.add_argument(
        "--metrics-out", default="artifacts/reports/shanghaitech_70_30_metrics.json", help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--manifest-out", default="data/interim/shanghaitech_70_30_manifest.csv", help="Path to save split manifest CSV"
    )
    return parser.parse_args()


def _tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tolerance = np.maximum(20.0, 0.2 * y_true)
    within = np.abs(y_pred - y_true) <= tolerance
    return float(np.mean(within))


def _save_manifest(
    ids: list[str], parts: list[str], targets: np.ndarray, train_idx: list[int], test_idx: list[int], out_csv: Path
) -> None:
    train_set = set(train_idx)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "part", "count", "split"])
        for idx, sample_id in enumerate(ids):
            split = "train" if idx in train_set else "test"
            writer.writerow([sample_id, parts[idx], float(targets[idx]), split])


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    samples = collect_samples(dataset_root)
    x, y, parts, ids = load_features_and_targets(samples, image_size=(args.image_size, args.image_size))
    train_idx, test_idx = split_indices(parts, train_ratio=args.train_ratio, seed=args.seed)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).astype(np.float32)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    tol_acc = _tolerance_accuracy(y_test, y_pred)

    metrics = {
        "dataset_root": str(dataset_root),
        "samples_total": int(len(y)),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_ratio": float(len(train_idx) / len(y)),
        "test_ratio": float(len(test_idx) / len(y)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "tolerance_accuracy": tol_acc,
        "model": "RandomForestRegressor",
        "n_estimators": int(args.n_estimators),
        "seed": int(args.seed),
        "image_size": int(args.image_size),
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _save_manifest(ids, parts, y, train_idx, test_idx, out_csv=Path(args.manifest_out))

    print(f"train_samples={len(train_idx)} test_samples={len(test_idx)}")
    print(f"mae={mae:.4f} rmse={rmse:.4f} r2={r2:.4f} tolerance_accuracy={tol_acc:.4f}")
    print(f"metrics={metrics_out}")
    print(f"model={model_out}")
    print(f"manifest={Path(args.manifest_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
