from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

from src.data.shanghaitech_count import (
    collect_samples,
    extract_count_from_mat,
    load_features_and_targets,
    split_indices,
)


def _make_sample(root: Path, part: str, split: str, idx: int, count: int) -> None:
    image_dir = root / part / split / "images"
    gt_dir = root / part / split / "ground-truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"IMG_{idx}.jpg"
    frame = np.full((24, 32, 3), min(255, count), dtype=np.uint8)
    cv2.imwrite(str(image_path), frame)

    gt_path = gt_dir / f"GT_IMG_{idx}.mat"
    location = np.zeros((count, 2), dtype=np.float32)
    number = np.array([[count]], dtype=np.uint16)
    info = np.empty((1, 1), dtype=[("location", "O"), ("number", "O")])
    info[0, 0] = (location, number)
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = info
    savemat(gt_path, {"image_info": image_info})


def test_collect_and_extract_counts(tmp_path: Path) -> None:
    _make_sample(tmp_path, "part_A", "train_data", idx=1, count=12)
    _make_sample(tmp_path, "part_B", "test_data", idx=2, count=34)

    samples = collect_samples(tmp_path)

    assert len(samples) == 2
    counts = sorted(extract_count_from_mat(s.gt_path) for s in samples)
    assert counts == [12, 34]


def test_load_features_and_targets_shapes(tmp_path: Path) -> None:
    for idx in range(1, 6):
        _make_sample(tmp_path, "part_A", "train_data", idx=idx, count=idx * 10)

    samples = collect_samples(tmp_path)
    x, y, parts, ids = load_features_and_targets(samples, image_size=(16, 16))

    assert x.shape == (5, 256)
    assert y.shape == (5,)
    assert len(parts) == 5
    assert len(ids) == 5


def test_split_indices_respects_ratio_and_seed() -> None:
    parts = ["part_A"] * 70 + ["part_B"] * 30

    train_idx, test_idx = split_indices(parts, train_ratio=0.7, seed=42)
    train2, test2 = split_indices(parts, train_ratio=0.7, seed=42)

    assert len(train_idx) == 70
    assert len(test_idx) == 30
    assert train_idx == train2
    assert test_idx == test2
