from __future__ import annotations

import os
import random
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg"}


def collect_video_files(videos_dir: Path) -> list[Path]:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory does not exist: {videos_dir}")
    videos = [p for p in sorted(videos_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No video files found in: {videos_dir}")
    return videos


def split_video_files(videos: list[Path], train_ratio: float, seed: int = 42) -> list[tuple[Path, str]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if len(videos) < 2:
        raise ValueError("Need at least 2 videos to create train/test split")

    items = list(videos)
    rng = random.Random(seed)
    rng.shuffle(items)

    train_count = max(1, int(round(len(items) * train_ratio)))
    train_count = min(train_count, len(items) - 1)
    assignments: list[tuple[Path, str]] = []
    for idx, path in enumerate(items):
        split = "train" if idx < train_count else "test"
        assignments.append((path, split))
    return assignments


def materialize_split_links(assignments: list[tuple[Path, str]], output_root: Path) -> tuple[int, int]:
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    for source, split in assignments:
        target_dir = train_dir if split == "train" else test_dir
        target = target_dir / source.name
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

        if split == "train":
            train_count += 1
        else:
            test_count += 1
    return train_count, test_count
