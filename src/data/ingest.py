from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
import yaml

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg"}
VALID_SUBSETS = {"part_a", "part_b", "both"}


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    dataset_subset: str
    data_root: Path
    metadata_csv: Path

    def __post_init__(self) -> None:
        if self.dataset_subset not in VALID_SUBSETS:
            raise ValueError(
                f"dataset_subset must be one of {sorted(VALID_SUBSETS)}, got {self.dataset_subset!r}"
            )


def load_data_config(config_path: Path) -> DataConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = raw.get("dataset", {})
    paths = raw.get("paths", {})
    return DataConfig(
        dataset_name=str(dataset.get("name", "shanghaitech")),
        dataset_subset=str(dataset.get("subset", "part_a")),
        data_root=Path(paths.get("data_root", "data/raw")),
        metadata_csv=Path(paths.get("metadata_csv", "data/interim/metadata.csv")),
    )


def _selected_subsets(dataset_subset: str) -> Iterable[str]:
    if dataset_subset == "both":
        return ("part_a", "part_b")
    return (dataset_subset,)


def _probe_video(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames


def _has_label(split_root: Path, video_id: str) -> bool:
    annotation_dir = split_root / "annotations"
    if not annotation_dir.exists():
        return False
    for ext in (".txt", ".json", ".mat", ".csv", ".xml"):
        if (annotation_dir / f"{video_id}{ext}").exists():
            return True
    return False


def build_metadata(config: DataConfig) -> pd.DataFrame:
    dataset_root = config.data_root / config.dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rows: list[dict[str, object]] = []
    for subset in _selected_subsets(config.dataset_subset):
        subset_root = dataset_root / subset
        if not subset_root.exists():
            raise FileNotFoundError(f"Subset path not found: {subset_root}")

        for split in ("train", "val", "test"):
            split_root = subset_root / split
            if not split_root.exists():
                continue

            video_root = split_root / "videos"
            search_root = video_root if video_root.exists() else split_root
            for video_path in sorted(search_root.rglob("*")):
                if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                fps, frames = _probe_video(video_path)
                rows.append(
                    {
                        "video_id": video_path.stem,
                        "subset": subset,
                        "split": split,
                        "video_path": str(video_path),
                        "fps": fps,
                        "frames": frames,
                        "label_available": int(_has_label(split_root, video_path.stem)),
                    }
                )

    if not rows:
        raise FileNotFoundError("No video files found for selected dataset subset.")

    metadata = pd.DataFrame(rows)
    metadata = metadata.sort_values(["subset", "split", "video_id"]).reset_index(drop=True)
    return metadata


def write_metadata(config: DataConfig) -> Path:
    metadata = build_metadata(config)
    config.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(config.metadata_csv, index=False)
    return config.metadata_csv
