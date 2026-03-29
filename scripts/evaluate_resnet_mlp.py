# scripts/evaluate_resnet_mlp.py
"""Evaluate ResNet18+MLP on the ShanghaiTech test set.

Usage
-----
    python scripts/evaluate_resnet_mlp.py \
        [--frames-dir data/raw/shanghaitech/shanghaitech/testing/frames] \
        [--masks-dir  data/raw/shanghaitech/shanghaitech/testing/test_frame_mask] \
        [--model-path artifacts/models/resnet_mlp.pt]
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

from src.inference.resnet_mlp_model import load_resnet_mlp_model_fn

_DEFAULT_FRAMES = Path("data/raw/shanghaitech/shanghaitech/testing/frames")
_DEFAULT_MASKS  = Path("data/raw/shanghaitech/shanghaitech/testing/test_frame_mask")
_DEFAULT_MODEL  = Path("artifacts/models/resnet_mlp.pt")
_CLIP_LEN       = 30


def _load_clip_frames(frame_dir: Path) -> np.ndarray:
    paths  = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = [cv2.imread(str(p)) for p in paths]
    return np.stack(frames, axis=0)


def evaluate(args: argparse.Namespace) -> None:
    frames_root = Path(args.frames_dir)
    masks_root  = Path(args.masks_dir)

    anomaly_fn  = load_resnet_mlp_model_fn(model_path=Path(args.model_path))

    clip_dirs   = sorted(frames_root.iterdir())
    all_scores: list[float] = []
    all_labels: list[int]   = []

    for clip_dir in clip_dirs:
        mask_file = masks_root / f"{clip_dir.name}.npy"
        if not mask_file.exists():
            print(f"  [skip] no mask for {clip_dir.name}")
            continue

        frames = _load_clip_frames(clip_dir)
        mask   = np.load(str(mask_file))
        n      = len(frames)

        half         = _CLIP_LEN // 2
        frame_scores = np.zeros(n, dtype=np.float32)
        counts       = np.zeros(n, dtype=np.int32)

        for start in range(0, n - _CLIP_LEN + 1, max(1, _CLIP_LEN // 4)):
            end    = start + _CLIP_LEN
            score  = anomaly_fn(frames[start:end])
            centre = start + half
            frame_scores[centre] += score
            counts[centre]       += 1

        valid = counts > 0
        frame_scores[valid] /= counts[valid]
        last = 0.0
        for i in range(n):
            if counts[i] > 0:
                last = frame_scores[i]
            else:
                frame_scores[i] = last

        all_scores.extend(frame_scores.tolist())
        all_labels.extend(mask.astype(int).tolist())
        print(f"  {clip_dir.name}: {n} frames, anomaly%={mask.mean()*100:.1f}%")

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    roc_auc = roc_auc_score(labels_arr, scores_arr)
    pr_auc  = average_precision_score(labels_arr, scores_arr)

    print(f"\n=== ResNet18 + MLP Results ===")
    print(f"Clips evaluated : {len(clip_dirs)}")
    print(f"Total frames    : {len(labels_arr)}")
    print(f"ROC-AUC         : {roc_auc:.4f}  (RF baseline: 0.8313)")
    print(f"PR-AUC          : {pr_auc:.4f}  (RF baseline: 0.8261)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default=str(_DEFAULT_FRAMES))
    parser.add_argument("--masks-dir",  default=str(_DEFAULT_MASKS))
    parser.add_argument("--model-path", default=str(_DEFAULT_MODEL))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
