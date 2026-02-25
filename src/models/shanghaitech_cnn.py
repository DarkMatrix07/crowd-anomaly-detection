from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from src.data.shanghaitech_count import ShanghaiTechSample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    tolerance = np.maximum(20.0, 0.2 * y_true)
    within = np.abs(y_pred - y_true) <= tolerance
    return float(np.mean(within))


def build_model(model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    normalized = model_name.lower()
    if normalized == "tiny":
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    if normalized == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return model

    raise ValueError("model_name must be 'tiny' or 'resnet18'")


@dataclass(frozen=True)
class CountBatch:
    image: torch.Tensor
    count: torch.Tensor
    sample_id: str
    part: str


class ShanghaiTechCountDataset(Dataset):
    def __init__(
        self,
        samples: list[ShanghaiTechSample],
        indices: list[int],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.samples = [samples[i] for i in indices]
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unable to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
            jitter = np.random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx: int) -> CountBatch:
        sample = self.samples[idx]
        from src.data.shanghaitech_count import extract_count_from_mat  # local import to avoid circular cost

        image = self._load_image(sample.image_path).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        count = float(extract_count_from_mat(sample.gt_path))
        count_tensor = torch.tensor(np.log1p(count), dtype=torch.float32)
        return CountBatch(
            image=image_tensor,
            count=count_tensor,
            sample_id=sample.sample_id,
            part=sample.part,
        )


def collate_count_batches(batch: list[CountBatch]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    images = torch.stack([item.image for item in batch], dim=0)
    counts = torch.stack([item.count for item in batch], dim=0).unsqueeze(1)
    ids = [item.sample_id for item in batch]
    parts = [item.part for item in batch]
    return images, counts, ids, parts
