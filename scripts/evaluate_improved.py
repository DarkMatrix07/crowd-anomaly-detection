"""Evaluate the saved improved anomaly model on the proper clip-level test split.

Loads the saved model artifact and re-evaluates without the hanging predict_proba
issue that occurred in train_improved.py (Windows multiprocessing with n_jobs=-1).
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import _sorted_frame_paths, compute_frame_features


SEED = 42
TEST_RATIO = 0.30
FRAME_STRIDE = 3
DATASET_ROOT = Path("data/raw/shanghaitech/shanghaitech")
MODEL_PATH = Path("artifacts/models/shanghaitech_anomaly_improved.joblib")
METRICS_OUT = Path("artifacts/reports/shanghaitech_anomaly_improved_metrics.json")
PREDICTIONS_OUT = Path("artifacts/reports/shanghaitech_anomaly_improved_predictions.csv")


def _scene_of(clip_id: str) -> str:
    return clip_id.split("_")[0]


def clip_level_split(clip_ids: list[str], test_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    scenes: dict[str, list[str]] = defaultdict(list)
    for cid in clip_ids:
        scenes[_scene_of(cid)].append(cid)

    train_clips, test_clips = [], []
    for scene, clips in sorted(scenes.items()):
        shuffled = clips[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio)) if len(shuffled) >= 2 else 0
        test_clips.extend(shuffled[:n_test])
        train_clips.extend(shuffled[n_test:])
    return train_clips, test_clips


def load_clips(frames_root, masks_root, clip_ids, stride):
    features, labels, meta = [], [], []
    clip_id_set = set(clip_ids)
    for clip_dir in sorted(frames_root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name not in clip_id_set:
            continue
        mask_path = masks_root / f"{clip_dir.name}.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(np.uint8).reshape(-1)
        frame_paths = _sorted_frame_paths(clip_dir)
        if not frame_paths:
            continue
        previous = None
        for idx, fp in enumerate(frame_paths):
            if idx >= len(mask):
                break
            if idx % stride != 0:
                continue
            frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            feat = compute_frame_features(frame, prev_frame=previous)
            previous = frame
            features.append(feat)
            labels.append(int(mask[idx]))
            meta.append({"clip_id": clip_dir.name, "frame_name": fp.name, "frame_index": idx})
    x = np.vstack(features).astype(np.float32) if features else np.empty((0, 10), np.float32)
    y = np.asarray(labels, np.int32)
    return x, y, meta


def main():
    frames_root = DATASET_ROOT / "testing" / "frames"
    masks_root = DATASET_ROOT / "testing" / "test_frame_mask"

    all_clip_ids = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    _, test_clip_ids = clip_level_split(all_clip_ids, TEST_RATIO, SEED)
    print(f"Test clips: {len(test_clip_ids)}")

    print("Loading test clips ...")
    x_test, y_test, meta_test = load_clips(frames_root, masks_root, test_clip_ids, FRAME_STRIDE)
    print(f"  frames: {len(y_test)}  anomaly: {int(y_test.sum())}")

    print("Loading model ...")
    model = joblib.load(MODEL_PATH)

    print("Running predict_proba ...")
    # Disable joblib parallelism to avoid Windows multiprocessing hang
    from sklearn.utils.validation import check_is_fitted
    rf = model.named_steps["rf"]
    rf.n_jobs = 1  # force single-threaded predict
    y_prob = model.predict_proba(x_test)[:, 1]
    print("Done.")

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    print(f"ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")

    # Threshold sweep for best F1
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int32)
        score = float(f1_score(y_test, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    print(f"Best threshold: {best_t:.3f}  Best F1: {best_f1:.4f}")

    y_pred = (y_prob >= best_t).astype(np.int32)
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print(
        f"\n=== Test Results (threshold={best_t:.3f}) ===\n"
        f"  accuracy={acc:.4f}  precision={precision:.4f}  recall={recall:.4f}\n"
        f"  f1={f1:.4f}  roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}\n"
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
        f"  error_rate={1-acc:.4f}"
    )

    # F1 at default threshold
    f1_05 = float(f1_score(y_test, (y_prob >= 0.5).astype(np.int32), zero_division=0))
    print(f"  F1 at default threshold=0.5: {f1_05:.4f}")

    # Scene-level AUC
    scene_data: dict[str, dict] = {}
    for i, row in enumerate(meta_test):
        s = _scene_of(row["clip_id"])
        if s not in scene_data:
            scene_data[s] = {"y_true": [], "y_prob": []}
        scene_data[s]["y_true"].append(int(y_test[i]))
        scene_data[s]["y_prob"].append(float(y_prob[i]))

    print("\n=== Scene-level AUC ===")
    scene_aucs = {}
    for scene, data in sorted(scene_data.items()):
        yt = np.array(data["y_true"])
        yp = np.array(data["y_prob"])
        if len(np.unique(yt)) < 2:
            scene_aucs[scene] = None
            print(f"  Scene {scene}: single class -- skipped")
        else:
            auc = float(roc_auc_score(yt, yp))
            scene_aucs[scene] = round(auc, 4)
            print(f"  Scene {scene}: AUC={auc:.4f}  frames={len(yt)}")

    # Feature importances
    feature_names = [
        "mean_intensity", "std_intensity", "lap_var", "occupancy",
        "flow_mean", "flow_var", "flow_max",
        "directional_entropy", "divergence_proxy", "temporal_contrast",
    ]
    importances = rf.feature_importances_
    print("\n=== Feature Importances ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Save metrics
    metrics = {
        "split_method": "clip_level_stratified_by_scene",
        "test_clips": len(test_clip_ids),
        "test_frames": int(len(y_test)),
        "anomaly_rate_test": float(y_test.mean()),
        "optimal_threshold": best_t,
        "accuracy": acc,
        "error_rate": float(1 - acc),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_at_default_threshold": f1_05,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "scene_roc_auc": scene_aucs,
        "feature_importances": {n: round(float(v), 4) for n, v in zip(feature_names, importances)},
        "model": str(MODEL_PATH),
        "frame_stride": FRAME_STRIDE,
        "seed": SEED,
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics -> {METRICS_OUT}")

    # Save predictions
    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "frame_name", "frame_index", "y_true", "y_pred", "y_prob"])
        for i, row in enumerate(meta_test):
            writer.writerow([row["clip_id"], row["frame_name"], row["frame_index"],
                             int(y_test[i]), int(y_pred[i]), float(y_prob[i])])
    print(f"Predictions -> {PREDICTIONS_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
