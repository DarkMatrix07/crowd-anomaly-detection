# scripts/evaluate_autoencoder.py
"""Evaluate the trained CNN autoencoder on the ShanghaiTech test set.

Computes frame-level ROC-AUC and PR-AUC to compare against the RF baseline.

Usage
-----
    python scripts/evaluate_autoencoder.py \
        [--frames-dir data/raw/shanghaitech/shanghaitech/testing/frames] \
        [--masks-dir data/raw/shanghaitech/shanghaitech/testing/test_frame_mask] \
        [--model-path artifacts/models/autoencoder.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn

_DEFAULT_FRAMES = Path("data/raw/shanghaitech/shanghaitech/testing/frames")
_DEFAULT_MASKS  = Path("data/raw/shanghaitech/shanghaitech/testing/test_frame_mask")
_DEFAULT_MODEL  = Path("artifacts/models/autoencoder.pt")

# Frames per clip sent to anomaly_model_fn (matches pipeline clip_length)
_CLIP_LEN_PIPELINE = 30


def _load_clip_frames(frame_dir: Path) -> np.ndarray:
    """Load all jpg frames from a clip directory as (N, H, W, 3) uint8 BGR."""
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = [cv2.imread(str(p)) for p in paths]
    return np.stack(frames, axis=0)


def evaluate(args: argparse.Namespace) -> None:
    frames_root = Path(args.frames_dir)
    masks_root  = Path(args.masks_dir)
    model_path  = Path(args.model_path)

    anomaly_fn = load_cnn_anomaly_model_fn(model_path=model_path)

    clip_dirs = sorted(frames_root.iterdir())
    all_scores: list[float] = []
    all_labels: list[int] = []

    for clip_dir in clip_dirs:
        mask_file = masks_root / f"{clip_dir.name}.npy"
        if not mask_file.exists():
            print(f"  [skip] no mask for {clip_dir.name}")
            continue

        frames = _load_clip_frames(clip_dir)        # (N, H, W, 3)
        mask   = np.load(str(mask_file))            # (N,) binary

        n = len(frames)
        # Score each window of _CLIP_LEN_PIPELINE frames, assign score to centre frame
        half = _CLIP_LEN_PIPELINE // 2
        frame_scores = np.zeros(n, dtype=np.float32)
        counts       = np.zeros(n, dtype=np.int32)

        for start in range(0, n - _CLIP_LEN_PIPELINE + 1, max(1, _CLIP_LEN_PIPELINE // 4)):
            end = start + _CLIP_LEN_PIPELINE
            clip = frames[start:end]
            score = anomaly_fn(clip)
            centre = start + half
            frame_scores[centre] += score
            counts[centre] += 1

        # Average overlapping window scores
        valid = counts > 0
        frame_scores[valid] /= counts[valid]
        # Forward-fill gaps at boundaries
        last = 0.0
        for i in range(n):
            if counts[i] > 0:
                last = frame_scores[i]
            else:
                frame_scores[i] = last

        all_scores.extend(frame_scores.tolist())
        all_labels.extend(mask.astype(int).tolist())
        print(f"  {clip_dir.name}: {n} frames, anomaly%={mask.mean()*100:.1f}%")

    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels_arr, all_scores_arr)
    pr_auc  = average_precision_score(all_labels_arr, all_scores_arr)

    print(f"\n=== CNN Autoencoder Results ===")
    print(f"Clips evaluated : {len(clip_dirs)}")
    print(f"Total frames    : {len(all_labels_arr)}")
    print(f"ROC-AUC         : {roc_auc:.4f}  (RF baseline: 0.8313)")
    print(f"PR-AUC          : {pr_auc:.4f}  (RF baseline: 0.8261)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN autoencoder on ShanghaiTech test set")
    parser.add_argument("--frames-dir", default=str(_DEFAULT_FRAMES))
    parser.add_argument("--masks-dir",  default=str(_DEFAULT_MASKS))
    parser.add_argument("--model-path", default=str(_DEFAULT_MODEL))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
