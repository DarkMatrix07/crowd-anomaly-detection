from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import collect_video_files, materialize_split_links, split_video_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible 70/30 train-test split for ShanghaiTech videos")
    parser.add_argument(
        "--videos-dir",
        default="data/raw/shanghaitech/shanghaitech/training/videos",
        help="Directory containing ShanghaiTech training videos",
    )
    parser.add_argument(
        "--out-csv",
        default="data/interim/shanghaitech_train_test_70_30.csv",
        help="CSV path for split manifest",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/splits/shanghaitech_70_30",
        help="Directory to materialize train/test hardlinks",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    out_csv = Path(args.out_csv)
    output_root = Path(args.output_root)

    videos = collect_video_files(videos_dir)
    assignments = split_video_files(videos, train_ratio=args.train_ratio, seed=args.seed)
    train_count, test_count = materialize_split_links(assignments, output_root=output_root)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_path", "split"])
        for path, split in assignments:
            writer.writerow([path.stem, str(path), split])

    total = train_count + test_count
    print(f"manifest={out_csv}")
    print(f"output_root={output_root}")
    print(f"train_videos={train_count}")
    print(f"test_videos={test_count}")
    print(f"ratio={train_count / total:.4f}/{test_count / total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
