from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import compute_frame_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained anomaly model on a single ShanghaiTech clip")
    parser.add_argument(
        "--model-path",
        default="artifacts/models/shanghaitech_anomaly_rf_70_30.joblib",
        help="Path to trained anomaly model",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/raw/shanghaitech/shanghaitech",
        help="ShanghaiTech root directory",
    )
    parser.add_argument("--clip-id", default="01_0014", help="Testing clip folder name under testing/frames/")
    parser.add_argument("--frame-stride", type=int, default=3, help="Evaluate every Nth frame")
    parser.add_argument("--max-frames", type=int, default=180, help="Max raw frames to process")
    parser.add_argument(
        "--out-csv",
        default="artifacts/reports/demo_clip_predictions.csv",
        help="CSV output path for per-frame predictions",
    )
    return parser.parse_args()


def _sorted_frames(clip_dir: Path) -> list[Path]:
    return sorted((p for p in clip_dir.glob("*.jpg") if p.is_file()), key=lambda p: int(p.stem))


def main() -> int:
    args = parse_args()
    model = joblib.load(Path(args.model_path))
    clip_dir = Path(args.dataset_root) / "testing" / "frames" / args.clip_id
    mask_path = Path(args.dataset_root) / "testing" / "test_frame_mask" / f"{args.clip_id}.npy"

    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip folder not found: {clip_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    frame_paths = _sorted_frames(clip_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {clip_dir}")

    feats: list[np.ndarray] = []
    frame_idx: list[int] = []
    previous = None
    for idx, path in enumerate(frame_paths[: args.max_frames]):
        if idx % args.frame_stride != 0:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        feats.append(compute_frame_features(image, prev_frame=previous))
        frame_idx.append(idx)
        previous = image

    x = np.vstack(feats).astype(np.float32)
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= 0.5).astype(np.uint8)

    true_mask = np.load(mask_path).astype(np.uint8).reshape(-1)
    y_true = np.array([true_mask[i] for i in frame_idx], dtype=np.uint8)
    accuracy = float(np.mean(pred == y_true))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "frame_index", "y_true", "y_pred", "y_prob"])
        for i, fidx in enumerate(frame_idx):
            writer.writerow([args.clip_id, fidx, int(y_true[i]), int(pred[i]), float(prob[i])])

    top = np.argsort(-prob)[:10]
    top_frames = [int(frame_idx[i]) for i in top]
    print(f"clip={args.clip_id}")
    print(f"samples={len(frame_idx)}")
    print(f"abnormal_pred={int(pred.sum())}")
    print(f"abnormal_true={int(y_true.sum())}")
    print(f"accuracy={accuracy:.4f}")
    print(f"top10_abnormal_frame_idx={top_frames}")
    print(f"csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
