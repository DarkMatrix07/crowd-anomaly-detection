from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.ingest import DataConfig, build_metadata, load_data_config


def _write_tiny_video(path: Path, frames: int = 6, fps: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create test video at {path}")
    for _ in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_dataset(root: Path) -> None:
    dataset_root = root / "shanghaitech"
    _write_tiny_video(dataset_root / "part_a" / "train" / "videos" / "a1.avi")
    _write_tiny_video(dataset_root / "part_a" / "test" / "videos" / "a2.avi")
    _write_tiny_video(dataset_root / "part_b" / "train" / "videos" / "b1.avi")
    (dataset_root / "part_a" / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "part_a" / "train" / "annotations" / "a1.txt").write_text("1\n", encoding="utf-8")


def test_load_data_config_reads_subset_and_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: shanghaitech",
                "  subset: both",
                "paths:",
                f"  data_root: {tmp_path.as_posix()}",
                f"  metadata_csv: {(tmp_path / 'metadata.csv').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert config.dataset_name == "shanghaitech"
    assert config.dataset_subset == "both"
    assert config.data_root == tmp_path
    assert config.metadata_csv == tmp_path / "metadata.csv"


def test_build_metadata_honors_subset_and_counts(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    config_a = DataConfig(
        dataset_name="shanghaitech",
        dataset_subset="part_a",
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata_a.csv",
    )
    config_both = DataConfig(
        dataset_name="shanghaitech",
        dataset_subset="both",
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata_both.csv",
    )

    meta_a = build_metadata(config_a)
    meta_both = build_metadata(config_both)

    assert len(meta_a) == 2
    assert set(meta_a["subset"]) == {"part_a"}
    assert len(meta_both) == 3
    assert set(meta_both["subset"]) == {"part_a", "part_b"}
    assert int(meta_a.loc[meta_a["video_id"] == "a1", "label_available"].iloc[0]) == 1


def test_build_metadata_raises_for_missing_subset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "shanghaitech" / "part_a" / "train" / "videos"
    _write_tiny_video(dataset_root / "only_a.avi")
    config = DataConfig(
        dataset_name="shanghaitech",
        dataset_subset="part_b",
        data_root=tmp_path,
        metadata_csv=tmp_path / "metadata.csv",
    )

    with pytest.raises(FileNotFoundError):
        build_metadata(config)
