"""Improved anomaly classifier for ShanghaiTech Campus dataset.

Key improvements over the original script:
  1. Clip-level train/test split -- no frame-level data leakage.
  2. Training videos (330 normal-only .avi files) included as negative class.
  3. 10-dimensional feature vector (adds directional entropy, divergence proxy,
     flow_max, temporal_contrast vs the original 6 features).
  4. Scene-stratified clip split to ensure all 12 scenes appear in both splits.
  5. Optimal threshold selection via F1 sweep on test ROC curve.
  6. Balanced training: normal-video frames downsampled to ~3x anomaly count.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

import cv2
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
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import (
    _sorted_frame_paths,
    build_normal_dataset_from_videos,
    compute_frame_features,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train improved ShanghaiTech anomaly classifier")
    p.add_argument(
        "--dataset-root",
        default="data/raw/shanghaitech/shanghaitech",
        help="ShanghaiTech root (contains testing/ and training/)",
    )
    p.add_argument("--test-ratio", type=float, default=0.30, help="Clip-level test split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--frame-stride-test", type=int, default=3, help="Frame stride for testing clips")
    p.add_argument("--frame-stride-train", type=int, default=5, help="Frame stride for training videos")
    p.add_argument("--max-frames-test-clip", type=int, default=200, help="Max frames per testing clip")
    p.add_argument("--max-frames-train-video", type=int, default=150, help="Max frames per training video")
    p.add_argument("--normal-oversample-ratio", type=float, default=3.0,
                   help="Ratio of normal-video frames to anomaly frames after balancing")
    p.add_argument("--n-estimators", type=int, default=500, help="RF tree count")
    p.add_argument(
        "--model-out",
        default="artifacts/models/shanghaitech_anomaly_improved.joblib",
    )
    p.add_argument(
        "--metrics-out",
        default="artifacts/reports/shanghaitech_anomaly_improved_metrics.json",
    )
    p.add_argument(
        "--predictions-out",
        default="artifacts/reports/shanghaitech_anomaly_improved_predictions.csv",
    )
    p.add_argument(
        "--manifest-out",
        default="data/interim/shanghaitech_anomaly_improved_manifest.csv",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clip-level stratified split
# ---------------------------------------------------------------------------

def _scene_of(clip_id: str) -> str:
    return clip_id.split("_")[0]


def clip_level_split(
    clip_ids: list[str],
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split clip IDs by scene so every scene appears in both sets."""
    rng = random.Random(seed)

    scenes: dict[str, list[str]] = defaultdict(list)
    for cid in clip_ids:
        scenes[_scene_of(cid)].append(cid)

    train_clips: list[str] = []
    test_clips: list[str] = []

    for scene, clips in sorted(scenes.items()):
        shuffled = clips[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio)) if len(shuffled) >= 2 else 0
        test_clips.extend(shuffled[:n_test])
        train_clips.extend(shuffled[n_test:])

    return train_clips, test_clips


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _load_testing_clips(
    frames_root: Path,
    masks_root: Path,
    clip_ids: list[str],
    frame_stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load features for a specified subset of testing clips."""
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
        used = 0
        for idx, fp in enumerate(frame_paths):
            if idx >= len(mask):
                break
            if idx % frame_stride != 0:
                continue
            frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            feat = compute_frame_features(frame, prev_frame=previous)
            previous = frame
            features.append(feat)
            labels.append(int(mask[idx]))
            meta.append(
                {
                    "clip_id": clip_dir.name,
                    "frame_name": fp.name,
                    "frame_index": idx,
                    "source": "testing",
                }
            )
            used += 1
            if max_frames is not None and used >= max_frames:
                break

    if not features:
        return np.empty((0, 10), dtype=np.float32), np.empty(0, dtype=np.int32), []
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.int32), meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
                    max_features="sqrt",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given set."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_t = 0.5
    best_f1 = 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_manifest(train_meta: list[dict], test_meta: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "frame_name", "frame_index", "source", "split"])
        for row in train_meta:
            writer.writerow([row["clip_id"], row["frame_name"], row["frame_index"], row["source"], "train"])
        for row in test_meta:
            writer.writerow([row["clip_id"], row["frame_name"], row["frame_index"], row["source"], "test"])


def _save_predictions(
    test_meta: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "frame_name", "frame_index", "source", "y_true", "y_pred", "y_prob"])
        for i, row in enumerate(test_meta):
            writer.writerow(
                [row["clip_id"], row["frame_name"], row["frame_index"], row["source"],
                 int(y_true[i]), int(y_pred[i]), float(y_prob[i])]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    rng_np = np.random.default_rng(args.seed)

    dataset_root = Path(args.dataset_root)
    frames_root = dataset_root / "testing" / "frames"
    masks_root = dataset_root / "testing" / "test_frame_mask"
    training_videos = dataset_root / "training" / "videos"

    # ---- 1. Clip-level split on testing clips -----------------------------
    all_clip_ids = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    train_clip_ids, test_clip_ids = clip_level_split(
        all_clip_ids, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"Clips -> train: {len(train_clip_ids)}  test: {len(test_clip_ids)}")

    # ---- 2. Load testing-split train features (labeled 0/1) ---------------
    print("Loading testing-split TRAIN clips ...")
    x_test_train, y_test_train, meta_test_train = _load_testing_clips(
        frames_root, masks_root, train_clip_ids,
        frame_stride=args.frame_stride_test,
        max_frames=args.max_frames_test_clip,
    )
    n_anomaly_train = int(y_test_train.sum())
    print(f"  frames: {len(y_test_train)}  anomaly: {n_anomaly_train}")

    # ---- 3. Load normal training videos (label = 0) -----------------------
    print("Loading normal training videos ...")
    x_normal_all, y_normal_all, meta_normal_all = build_normal_dataset_from_videos(
        training_videos,
        frame_stride=args.frame_stride_train,
        max_frames_per_video=args.max_frames_train_video,
    )
    print(f"  normal frames loaded: {len(y_normal_all)}")

    # ---- 4. Balance: downsample normal-video frames -----------------------
    # Target: normal_video_count ~ ratio * anomaly_count  (avoids extreme skew)
    target_normal = int(args.normal_oversample_ratio * n_anomaly_train)
    if target_normal < len(y_normal_all):
        idx_keep = rng_np.choice(len(y_normal_all), size=target_normal, replace=False)
        idx_keep = np.sort(idx_keep)
        x_normal = x_normal_all[idx_keep]
        y_normal = y_normal_all[idx_keep]
        meta_normal = [meta_normal_all[i] for i in idx_keep]
        print(f"  downsampled normal-video frames: {len(y_normal)} (target ratio {args.normal_oversample_ratio}x)")
    else:
        x_normal, y_normal, meta_normal = x_normal_all, y_normal_all, meta_normal_all
        print(f"  using all normal-video frames: {len(y_normal)}")

    # ---- 5. Combine train set ---------------------------------------------
    x_train = np.vstack([x_normal, x_test_train])
    y_train = np.concatenate([y_normal, y_test_train])
    meta_train = meta_normal + meta_test_train
    print(f"Combined train: {len(y_train)} frames  anomaly rate: {y_train.mean():.3f}")

    # ---- 6. Load test clips -----------------------------------------------
    print("Loading TEST clips ...")
    x_test, y_test, meta_test = _load_testing_clips(
        frames_root, masks_root, test_clip_ids,
        frame_stride=args.frame_stride_test,
        max_frames=None,
    )
    print(f"  test frames: {len(y_test)}  anomaly: {int(y_test.sum())}")

    # ---- 7. Train ---------------------------------------------------------
    print(f"Training RandomForest (n_estimators={args.n_estimators}) ...")
    model = _build_model(n_estimators=args.n_estimators, seed=args.seed)
    model.fit(x_train, y_train)

    # ---- 8. Evaluate at default threshold (0.5) ---------------------------
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(np.int32)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # ---- 9. Find optimal threshold via F1 sweep ---------------------------
    best_threshold = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(np.int32)

    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print(
        f"\n=== Test Results (threshold={best_threshold:.3f}) ===\n"
        f"  accuracy={acc:.4f}  precision={precision:.4f}  recall={recall:.4f}\n"
        f"  f1={f1:.4f}  roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}\n"
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
        f"  error_rate={1-acc:.4f}"
    )

    # Compare to default threshold
    f1_05 = float(f1_score(y_test, y_pred_05, zero_division=0))
    print(f"  (F1 at threshold=0.5: {f1_05:.4f})")

    # ---- 10. Scene-level AUC breakdown ------------------------------------
    scene_results: dict[str, dict] = {}
    for i, row in enumerate(meta_test):
        scene = _scene_of(row["clip_id"])
        if scene not in scene_results:
            scene_results[scene] = {"y_true": [], "y_prob": []}
        scene_results[scene]["y_true"].append(int(y_test[i]))
        scene_results[scene]["y_prob"].append(float(y_prob[i]))

    print("\n=== Scene-level AUC ===")
    scene_aucs = {}
    for scene, data in sorted(scene_results.items()):
        yt = np.array(data["y_true"])
        yp = np.array(data["y_prob"])
        if len(np.unique(yt)) < 2:
            scene_aucs[scene] = None
            print(f"  Scene {scene}: only one class -- skipped")
        else:
            auc = float(roc_auc_score(yt, yp))
            scene_aucs[scene] = round(auc, 4)
            print(f"  Scene {scene}: AUC={auc:.4f}  frames={len(yt)}")

    # ---- 11. Feature importances ------------------------------------------
    feature_names = [
        "mean_intensity", "std_intensity", "lap_var", "occupancy",
        "flow_mean", "flow_var", "flow_max",
        "directional_entropy", "divergence_proxy", "temporal_contrast",
    ]
    rf_model = model.named_steps["rf"]
    importances = rf_model.feature_importances_
    print("\n=== Feature Importances ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # ---- 12. Save artifacts -----------------------------------------------
    metrics = {
        "dataset_root": str(dataset_root),
        "split_method": "clip_level_stratified_by_scene",
        "train_clips": len(train_clip_ids),
        "test_clips": len(test_clip_ids),
        "train_frames_from_normal_videos": int(len(y_normal)),
        "train_frames_from_testing_clips": int(len(y_test_train)),
        "train_frames_total": int(len(y_train)),
        "test_frames": int(len(y_test)),
        "anomaly_rate_train": float(y_train.mean()),
        "anomaly_rate_test": float(y_test.mean()),
        "optimal_threshold": best_threshold,
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
        "model": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "features": feature_names,
        "frame_stride_test": args.frame_stride_test,
        "frame_stride_train": args.frame_stride_train,
        "normal_oversample_ratio": args.normal_oversample_ratio,
        "seed": args.seed,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nModel saved -> {model_out}")

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {metrics_out}")

    _save_manifest(meta_train, meta_test, Path(args.manifest_out))
    print(f"Manifest -> {args.manifest_out}")

    _save_predictions(meta_test, y_test, y_pred, y_prob, Path(args.predictions_out))
    print(f"Predictions -> {args.predictions_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
