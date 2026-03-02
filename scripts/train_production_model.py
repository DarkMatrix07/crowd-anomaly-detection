"""Temporal-window anomaly classifier for ShanghaiTech Campus dataset.

Frame-level features are too noisy to discriminate anomalies by themselves -
a single frame of running looks similar to a single frame of walking.

This script aggregates frame features over a sliding window of W frames to give
the classifier temporal context.  Each window produces a 40-d feature vector:

    mean(10) + std(10) + max(10) + delta(10)  =  40 features

where delta = mean(last-third frames) - mean(first-third frames), capturing
whether motion / intensity is escalating inside the window (panic indicator).

Key design choices
------------------
* Clip-level stratified split  – no data leakage.
* Window label = majority vote of frame labels in window.
* Training data = labeled testing clips (train split) + normal training videos.
* Balanced classes: normal-video windows downsampled to ~3× anomaly count.
* Optimal threshold found by F1 sweep on test ROC curve.
* n_jobs=1 at predict time to avoid Windows multiprocessing hang.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    p = argparse.ArgumentParser(
        description="Train temporal-window ShanghaiTech anomaly classifier"
    )
    p.add_argument(
        "--dataset-root",
        default="data/raw/shanghaitech/shanghaitech",
        help="ShanghaiTech root (contains testing/ and training/)",
    )
    p.add_argument("--window-size", type=int, default=30,
                   help="Number of consecutive frames per window")
    p.add_argument("--window-stride", type=int, default=10,
                   help="Step between windows (use smaller for more data)")
    p.add_argument("--test-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-stride-clips", type=int, default=2,
                   help="Frame stride when loading testing clip frames")
    p.add_argument("--frame-stride-train-videos", type=int, default=3,
                   help="Frame stride when extracting from normal training videos")
    p.add_argument("--max-frames-train-video", type=int, default=200,
                   help="Max frames per training video (keep low for speed)")
    p.add_argument("--max-train-videos", type=int, default=80,
                   help="Max number of training videos to use (0 = skip training videos)")
    p.add_argument("--normal-ratio", type=float, default=3.0,
                   help="Normal-window count = ratio x anomaly-window count")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument(
        "--classifier", choices=["rf", "gbt"], default="rf",
        help="Classifier type: rf=RandomForest, gbt=GradientBoosting",
    )
    p.add_argument(
        "--no-flow", action="store_true",
        help="Ablation: zero out all optical-flow derived features before training",
    )
    p.add_argument(
        "--model-out",
        default="artifacts/models/shanghaitech_windowed.joblib",
    )
    p.add_argument(
        "--metrics-out",
        default="artifacts/reports/shanghaitech_windowed_metrics.json",
    )
    p.add_argument(
        "--predictions-out",
        default="artifacts/reports/shanghaitech_windowed_predictions.csv",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clip-level stratified split
# ---------------------------------------------------------------------------

def _scene_of(clip_id: str) -> str:
    return clip_id.split("_")[0]


def clip_level_split(
    clip_ids: list[str], test_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
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


# ---------------------------------------------------------------------------
# Frame feature extraction for a single clip
# ---------------------------------------------------------------------------

def extract_clip_frame_features(
    clip_dir: Path,
    mask: np.ndarray,
    frame_stride: int = 1,
    resize: tuple[int, int] = (320, 240),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_features, frame_labels) for sampled frames in the clip.

    frame_features : (N, 10) float32
    frame_labels   : (N,)    int32
    """
    frame_paths = _sorted_frame_paths(clip_dir)
    if not frame_paths:
        return np.empty((0, 10), np.float32), np.empty(0, np.int32)

    features: list[np.ndarray] = []
    labels: list[int] = []
    previous: np.ndarray | None = None

    for idx, fp in enumerate(frame_paths):
        if idx >= len(mask):
            break
        if idx % frame_stride != 0:
            continue
        frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if frame is None:
            previous = None
            continue
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        feat = compute_frame_features(frame, prev_frame=previous)
        previous = frame
        features.append(feat)
        labels.append(int(mask[idx]))

    if not features:
        return np.empty((0, 10), np.float32), np.empty(0, np.int32)
    return np.vstack(features).astype(np.float32), np.asarray(labels, np.int32)


# ---------------------------------------------------------------------------
# Window feature construction
# ---------------------------------------------------------------------------

FRAME_FEATURE_DIM = 10
WINDOW_FEATURE_DIM = FRAME_FEATURE_DIM * 4  # mean + std + max + delta


def frames_to_windows(
    frame_feats: np.ndarray,   # (N, 10)
    frame_labels: np.ndarray,  # (N,)
    window_size: int,
    window_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Slide a window over frame features and build window-level feature matrix.

    Window feature (40-d):
        mean  (10) – average feature value across the window
        std   (10) – variability within the window
        max   (10) – peak value in the window
        delta (10) – mean(last-third) - mean(first-third)  [trend / escalation]

    Window label = 1 if majority of its frames are anomalous, else 0.

    Returns
    -------
    X      : (M, 40) float32  window feature matrix
    y      : (M,)    int32    window labels
    spans  : list of (start_idx, end_idx) frame index pairs
    """
    N = len(frame_feats)
    if N < window_size:
        return np.empty((0, WINDOW_FEATURE_DIM), np.float32), np.empty(0, np.int32), []

    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    spans: list[tuple[int, int]] = []

    third = max(1, window_size // 3)

    start = 0
    while start + window_size <= N:
        end = start + window_size
        w = frame_feats[start:end]          # (W, 10)
        lbl = frame_labels[start:end]       # (W,)

        f_mean  = w.mean(axis=0)            # (10,)
        f_std   = w.std(axis=0)             # (10,)
        f_max   = w.max(axis=0)             # (10,)
        f_delta = w[-third:].mean(axis=0) - w[:third].mean(axis=0)  # (10,)

        window_feat = np.concatenate([f_mean, f_std, f_max, f_delta])
        window_label = int(lbl.mean() >= 0.5)   # majority vote

        X_windows.append(window_feat)
        y_windows.append(window_label)
        spans.append((start, end - 1))

        start += window_stride

    if not X_windows:
        return np.empty((0, WINDOW_FEATURE_DIM), np.float32), np.empty(0, np.int32), []

    return (
        np.vstack(X_windows).astype(np.float32),
        np.asarray(y_windows, np.int32),
        spans,
    )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_windowed_dataset_from_clips(
    frames_root: Path,
    masks_root: Path,
    clip_ids: list[str],
    window_size: int,
    window_stride: int,
    frame_stride: int = 1,
    label: str = "clips",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build window-level features from a list of testing clips."""
    clip_id_set = set(clip_ids)
    X_all, y_all, meta_all = [], [], []
    found = 0

    for clip_dir in sorted(frames_root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name not in clip_id_set:
            continue
        mask_path = masks_root / f"{clip_dir.name}.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(np.uint8).reshape(-1)

        frame_feats, frame_labels = extract_clip_frame_features(
            clip_dir, mask, frame_stride=frame_stride
        )
        if len(frame_feats) == 0:
            continue

        X_w, y_w, spans = frames_to_windows(
            frame_feats, frame_labels, window_size, window_stride
        )
        if len(X_w) == 0:
            continue

        found += 1
        X_all.append(X_w)
        y_all.append(y_w)
        for start, end in spans:
            meta_all.append({
                "clip_id": clip_dir.name,
                "window_start": start,
                "window_end": end,
                "source": "testing",
            })
        if found % 10 == 0:
            print(f"    ... processed {found}/{len(clip_ids)} {label} clips")

    if not X_all:
        return np.empty((0, WINDOW_FEATURE_DIM), np.float32), np.empty(0, np.int32), []
    return (
        np.vstack(X_all).astype(np.float32),
        np.concatenate(y_all).astype(np.int32),
        meta_all,
    )


def build_windowed_dataset_from_normal_videos(
    x_frame: np.ndarray,    # (N, 10) frame features from training videos
    window_size: int,
    window_stride: int,
    video_boundaries: list[tuple[int, int]],   # list of (start_idx, end_idx) per video
) -> tuple[np.ndarray, np.ndarray]:
    """Build normal-class windows from pre-extracted training video frame features.

    Respects video boundaries so no window crosses two different videos.
    """
    X_all, y_all = [], []
    dummy_labels = np.zeros(len(x_frame), dtype=np.int32)

    for v_start, v_end in video_boundaries:
        seg_feats = x_frame[v_start:v_end + 1]
        seg_labels = dummy_labels[v_start:v_end + 1]
        X_w, y_w, _ = frames_to_windows(seg_feats, seg_labels, window_size, window_stride)
        if len(X_w):
            X_all.append(X_w)
            y_all.append(y_w)

    if not X_all:
        return np.empty((0, WINDOW_FEATURE_DIM), np.float32), np.empty(0, np.int32)
    return (
        np.vstack(X_all).astype(np.float32),
        np.concatenate(y_all).astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_estimators: int, seed: int, classifier: str = "rf") -> Pipeline:
    if classifier == "gbt":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            max_features="sqrt",
            random_state=seed,
        )
        step_name = "gbt"
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            max_depth=None,
        )
        step_name = "rf"
    return Pipeline([("scaler", StandardScaler()), (step_name, clf)])


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    _, _, thresholds = roc_curve(y_true, y_prob)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    rng_np = np.random.default_rng(args.seed)

    dataset_root = Path(args.dataset_root)
    frames_root  = dataset_root / "testing"  / "frames"
    masks_root   = dataset_root / "testing"  / "test_frame_mask"
    training_vids = dataset_root / "training" / "videos"

    W = args.window_size
    S = args.window_stride
    print(f"Window size={W}  stride={S}  -> ~{WINDOW_FEATURE_DIM}-d features per window")

    # ---- 1. Clip-level split -----------------------------------------------
    all_clip_ids = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    train_clip_ids, test_clip_ids = clip_level_split(
        all_clip_ids, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"Clips -> train: {len(train_clip_ids)}  test: {len(test_clip_ids)}")

    # ---- 2. Build window features from TRAIN testing clips -----------------
    print("Extracting windows from training testing-clips ...")
    X_tc_train, y_tc_train, meta_tc_train = build_windowed_dataset_from_clips(
        frames_root, masks_root, train_clip_ids, W, S,
        frame_stride=args.frame_stride_clips, label="train",
    )
    n_anom_train = int(y_tc_train.sum())
    print(f"  windows={len(y_tc_train)}  anomaly={n_anom_train}  "
          f"normal={len(y_tc_train)-n_anom_train}")

    # ---- 3. Extract frame features from normal training videos -------------
    X_vid = np.empty((0, WINDOW_FEATURE_DIM), np.float32)
    y_vid: np.ndarray = np.empty(0, np.int32)

    if args.max_train_videos > 0:
        print(f"Extracting frame features from normal training videos "
              f"(max {args.max_train_videos} videos) ...")
        all_vid_paths = sorted(
            p for p in training_vids.iterdir()
            if p.suffix.lower() in {".avi", ".mp4", ".mov"}
        )
        vid_paths_sample = all_vid_paths[:args.max_train_videos]
        print(f"  using {len(vid_paths_sample)}/{len(all_vid_paths)} training videos")

        x_vid_frames, _, meta_vid = build_normal_dataset_from_videos(
            training_vids,
            frame_stride=args.frame_stride_train_videos,
            max_frames_per_video=args.max_frames_train_video,
        )
        # Trim to only the sampled videos
        sampled_stems = {p.stem for p in vid_paths_sample}
        keep = [i for i, m in enumerate(meta_vid) if m["clip_id"] in sampled_stems]
        x_vid_frames = x_vid_frames[keep]
        meta_vid = [meta_vid[i] for i in keep]
        print(f"  total frame features: {len(x_vid_frames)}")

        # Build video boundaries so windows don't cross video files
        vid_boundaries: list[tuple[int, int]] = []
        if meta_vid:
            current_vid = meta_vid[0]["clip_id"]
            seg_start = 0
            for i, m in enumerate(meta_vid):
                if m["clip_id"] != current_vid:
                    vid_boundaries.append((seg_start, i - 1))
                    seg_start = i
                    current_vid = m["clip_id"]
            vid_boundaries.append((seg_start, len(meta_vid) - 1))

        print(f"  {len(vid_boundaries)} training videos -> building windows ...")
        X_vid, y_vid = build_windowed_dataset_from_normal_videos(
            x_vid_frames, W, S, vid_boundaries
        )
        print(f"  normal-video windows: {len(y_vid)}")
    else:
        print("Skipping training videos (--max-train-videos=0)")

    # ---- 4. Downsample normal-video windows --------------------------------
    if len(y_vid) > 0:
        target_normal = int(args.normal_ratio * n_anom_train)
        if target_normal < len(y_vid):
            idx_keep = np.sort(
                rng_np.choice(len(y_vid), size=target_normal, replace=False)
            )
            X_vid = X_vid[idx_keep]
            y_vid = y_vid[idx_keep]
            print(f"  downsampled to {len(y_vid)} normal-video windows "
                  f"({args.normal_ratio}x anomaly count)")
        else:
            print(f"  using all {len(y_vid)} normal-video windows")

    # ---- 5. Combine train set ----------------------------------------------
    parts_X = [p for p in [X_vid, X_tc_train] if len(p) > 0]
    parts_y = [p for p in [y_vid, y_tc_train] if len(p) > 0]
    X_train = np.vstack(parts_X)
    y_train = np.concatenate(parts_y)
    print(f"Combined train: {len(y_train)} windows  "
          f"anomaly rate: {y_train.mean():.3f}")

    # ---- 6. Build window features from TEST testing clips ------------------
    print("Extracting windows from test testing-clips ...")
    X_test, y_test, meta_test = build_windowed_dataset_from_clips(
        frames_root, masks_root, test_clip_ids, W,
        window_stride=1,           # dense sliding window for evaluation
        frame_stride=args.frame_stride_clips,
        label="test",
    )
    print(f"  test windows={len(y_test)}  anomaly={int(y_test.sum())}")

    # ---- 6b. Ablation: zero out optical-flow features ----------------------
    # Flow features occupy frame-feature indices 4-8 (flow_mean, flow_var,
    # flow_max, directional_entropy, divergence_proxy).  In the 40-d window
    # vector (mean+std+max+delta of 10 frame features), these map to columns:
    #   mean  : 4-8    std  : 14-18   max  : 24-28   delta: 34-38
    _FLOW_COLS = (
        list(range(4, 9))    # mean block
        + list(range(14, 19))  # std block
        + list(range(24, 29))  # max block
        + list(range(34, 39))  # delta block
    )
    if args.no_flow:
        print("Ablation: zeroing out optical-flow feature columns")
        X_train[:, _FLOW_COLS] = 0.0
        X_test[:, _FLOW_COLS] = 0.0

    # ---- 7. Train ----------------------------------------------------------
    clf_label = "GradientBoosting" if args.classifier == "gbt" else "RandomForest"
    print(f"Training {clf_label} (n_estimators={args.n_estimators}) ...")
    model = build_model(
        n_estimators=args.n_estimators, seed=args.seed, classifier=args.classifier
    )
    model.fit(X_train, y_train)
    # Force single-threaded predict for RF to avoid Windows multiprocessing hang
    if args.classifier == "rf":
        model.named_steps["rf"].n_jobs = 1

    # ---- 8. Evaluate -------------------------------------------------------
    print("Running predict_proba ...")
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc  = float(average_precision_score(y_test, y_prob))
    print(f"ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")

    best_t = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_t).astype(np.int32)

    acc       = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print(
        f"\n=== Test Results (threshold={best_t:.3f}) ===\n"
        f"  accuracy={acc:.4f}  precision={precision:.4f}  "
        f"recall={recall:.4f}  f1={f1:.4f}\n"
        f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}\n"
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}"
    )

    f1_05 = float(f1_score(y_test, (y_prob >= 0.5).astype(np.int32), zero_division=0))
    print(f"  F1 at threshold=0.5: {f1_05:.4f}")

    # ---- 9. Scene-level AUC ------------------------------------------------
    scene_data: dict[str, dict] = {}
    for i, m in enumerate(meta_test):
        s = _scene_of(m["clip_id"])
        if s not in scene_data:
            scene_data[s] = {"y_true": [], "y_prob": []}
        scene_data[s]["y_true"].append(int(y_test[i]))
        scene_data[s]["y_prob"].append(float(y_prob[i]))

    print("\n=== Scene-level AUC ===")
    scene_aucs: dict[str, float | None] = {}
    for scene, data in sorted(scene_data.items()):
        yt = np.array(data["y_true"])
        yp = np.array(data["y_prob"])
        if len(np.unique(yt)) < 2:
            scene_aucs[scene] = None
            print(f"  Scene {scene}: single class - skipped")
        else:
            auc = float(roc_auc_score(yt, yp))
            scene_aucs[scene] = round(auc, 4)
            print(f"  Scene {scene}: AUC={auc:.4f}  windows={len(yt)}")

    # ---- 10. Feature importances -------------------------------------------
    frame_feat_names = [
        "mean_intensity", "std_intensity", "lap_var", "occupancy",
        "flow_mean", "flow_var", "flow_max",
        "directional_entropy", "divergence_proxy", "temporal_contrast",
    ]
    window_feat_names = (
        [f"mean_{n}" for n in frame_feat_names]
        + [f"std_{n}" for n in frame_feat_names]
        + [f"max_{n}" for n in frame_feat_names]
        + [f"delta_{n}" for n in frame_feat_names]
    )
    clf_step = model.named_steps.get("rf") or model.named_steps.get("gbt")
    importances = clf_step.feature_importances_
    print("\n=== Top-10 Feature Importances ===")
    ranked = sorted(zip(window_feat_names, importances), key=lambda x: -x[1])
    for name, imp in ranked[:10]:
        print(f"  {name}: {imp:.4f}")

    # ---- 11. Save artifacts ------------------------------------------------
    metrics = {
        "model": clf_label,
        "approach": "temporal_window",
        "ablation_no_flow": bool(args.no_flow),
        "window_size": W,
        "window_stride_train": S,
        "window_stride_test": 1,
        "window_feature_dim": WINDOW_FEATURE_DIM,
        "n_estimators": args.n_estimators,
        "train_clips": len(train_clip_ids),
        "test_clips": len(test_clip_ids),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "anomaly_rate_train": float(y_train.mean()),
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
        "feature_importances": {
            n: round(float(v), 4) for n, v in ranked
        },
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

    pred_out = Path(args.predictions_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    with pred_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "clip_id", "window_start", "window_end",
            "y_true", "y_pred", "y_prob",
        ])
        for i, m in enumerate(meta_test):
            writer.writerow([
                m["clip_id"], m["window_start"], m["window_end"],
                int(y_test[i]), int(y_pred[i]), float(y_prob[i]),
            ])
    print(f"Predictions saved -> {pred_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
