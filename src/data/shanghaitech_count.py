from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ShanghaiTechSample:
    part: str
    subset: str
    image_path: Path
    gt_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.part}/{self.subset}/{self.image_path.stem}"


def _iter_part_samples(dataset_root: Path, part: str) -> list[ShanghaiTechSample]:
    samples: list[ShanghaiTechSample] = []
    for subset in ("train_data", "test_data"):
        image_dir = dataset_root / part / subset / "images"
        gt_dir = dataset_root / part / subset / "ground-truth"
        if not image_dir.exists() or not gt_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("IMG_*.jpg")):
            idx = image_path.stem.split("_")[-1]
            gt_path = gt_dir / f"GT_IMG_{idx}.mat"
            if gt_path.exists():
                samples.append(
                    ShanghaiTechSample(
                        part=part,
                        subset=subset,
                        image_path=image_path,
                        gt_path=gt_path,
                    )
                )
    return samples


def collect_samples(dataset_root: Path) -> list[ShanghaiTechSample]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples = []
    for part in ("part_A", "part_B"):
        samples.extend(_iter_part_samples(root, part))
    if not samples:
        raise FileNotFoundError(f"No ShanghaiTech samples found under: {root}")
    return samples


def extract_count_from_mat(gt_path: Path) -> int:
    mat = loadmat(gt_path)
    entry = mat["image_info"][0, 0][0, 0]
    return int(entry["number"][0, 0])


def load_features_and_targets(
    samples: list[ShanghaiTechSample], image_size: tuple[int, int] = (32, 32)
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    width, height = image_size
    features: list[np.ndarray] = []
    targets: list[float] = []
    parts: list[str] = []
    ids: list[str] = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        vec = image.astype(np.float32).reshape(-1) / 255.0
        features.append(vec)
        targets.append(float(extract_count_from_mat(sample.gt_path)))
        parts.append(sample.part)
        ids.append(sample.sample_id)

    if not features:
        raise RuntimeError("Unable to load any images for features.")
    return (
        np.stack(features, axis=0),
        np.asarray(targets, dtype=np.float32),
        parts,
        ids,
    )


def split_indices(parts: list[str], train_ratio: float = 0.7, seed: int = 42) -> tuple[list[int], list[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    indices = list(range(len(parts)))
    train_size = int(round(train_ratio * len(indices)))
    train_size = max(1, min(train_size, len(indices) - 1))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        stratify=parts,
        shuffle=True,
    )
    return sorted(train_idx), sorted(test_idx)
