# scripts/train_resnet_mlp.py
"""Train a ResNet18 + MLP anomaly classifier on ShanghaiTech.

Strategy
--------
1. Frozen pretrained ResNet18 extracts 512-d features per frame.
2. Window aggregation (mean+std+max+delta over W frames) → 2048-d vector.
3. Small MLP trained with binary cross-entropy on normal (label=0) and
   anomalous (label=1) clips.

Normal clips  : training/videos  (all normal)
Anomaly clips : testing/frames   (frames where mask > 0 are anomalous)

Usage
-----
    python scripts/train_resnet_mlp.py \
        [--train-videos data/raw/shanghaitech/shanghaitech/training/videos] \
        [--test-frames  data/raw/shanghaitech/shanghaitech/testing/frames] \
        [--test-masks   data/raw/shanghaitech/shanghaitech/testing/test_frame_mask] \
        [--window-size 30] [--epochs 30] [--batch-size 64]
        [--out-path artifacts/models/resnet_mlp.pt]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

from src.models.resnet_mlp import WindowMLP

# ── Constants ────────────────────────────────────────────────────────────────
_FRAME_SIZE = (224, 224)   # ResNet18 expects 224×224
_FEAT_DIM   = 512          # ResNet18 avgpool output
_AGG_DIM    = _FEAT_DIM * 4  # mean+std+max+delta → 2048

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_DEFAULT_TRAIN  = Path("data/raw/shanghaitech/shanghaitech/training/videos")
_DEFAULT_FRAMES = Path("data/raw/shanghaitech/shanghaitech/testing/frames")
_DEFAULT_MASKS  = Path("data/raw/shanghaitech/shanghaitech/testing/test_frame_mask")
_DEFAULT_OUT    = Path("artifacts/models/resnet_mlp.pt")


# ── Feature extraction ────────────────────────────────────────────────────────

def _build_backbone(device: torch.device) -> torch.nn.Module:
    """Return frozen ResNet18 with avgpool as the final layer (outputs 512-d)."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()   # remove classifier head
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device).eval()


_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H,W,3) → normalised RGB tensor (3,224,224)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, _FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return _transform(rgb)


@torch.no_grad()
def _extract_clip_features(
    frames: np.ndarray,
    backbone: torch.nn.Module,
    device: torch.device,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """Extract window feature vectors from a clip.

    Returns a list of (2048,) float32 arrays, one per window.
    """
    n = len(frames)
    if n < window_size:
        return []

    # Extract per-frame ResNet features in one batch
    tensors = torch.stack([_frame_to_tensor(frames[i]) for i in range(n)]).to(device)
    # Process in sub-batches to avoid OOM
    feat_list = []
    for i in range(0, n, 64):
        feat_list.append(backbone(tensors[i:i+64]).cpu().numpy())
    frame_feats = np.concatenate(feat_list, axis=0)  # (N, 512)

    window_feats = []
    for start in range(0, n - window_size + 1, stride):
        w = frame_feats[start:start + window_size]   # (W, 512)
        third = max(1, window_size // 3)
        agg = np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.max(axis=0),
            w[-third:].mean(axis=0) - w[:third].mean(axis=0),
        ])  # (2048,)
        window_feats.append(agg.astype(np.float32))

    return window_feats


def _load_avi_frames(video_path: Path, max_frames: int = 300) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.stack(frames) if frames else np.empty((0, 1, 1, 3), dtype=np.uint8)


def _load_jpg_frames(frame_dir: Path) -> np.ndarray:
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = [cv2.imread(str(p)) for p in paths]
    return np.stack(frames) if frames else np.empty((0, 1, 1, 3), dtype=np.uint8)


# ── Dataset building ──────────────────────────────────────────────────────────

def build_dataset(
    train_videos_dir: Path,
    test_frames_dir: Path,
    test_masks_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for MLP training."""
    X_list, y_list = [], []

    # ── Normal clips (label=0) from training videos ───────────────────────
    print("Extracting features from normal training videos...")
    avi_files = sorted(train_videos_dir.glob("*.avi"))
    for vp in tqdm(avi_files, desc="Normal videos"):
        frames = _load_avi_frames(vp)
        if len(frames) < window_size:
            continue
        for feat in _extract_clip_features(frames, backbone, device, window_size, stride):
            X_list.append(feat)
            y_list.append(0)

    # ── Anomaly clips (label depends on mask) from test frames ───────────
    print("Extracting features from test clips...")
    clip_dirs = sorted(test_frames_dir.iterdir())
    for clip_dir in tqdm(clip_dirs, desc="Test clips"):
        mask_file = test_masks_dir / f"{clip_dir.name}.npy"
        if not mask_file.exists():
            continue
        frames = _load_jpg_frames(clip_dir)
        mask   = np.load(str(mask_file))   # (N,) binary
        if len(frames) < window_size:
            continue

        n = len(frames)
        for start in range(0, n - window_size + 1, stride):
            window_frames = frames[start:start + window_size]
            window_mask   = mask[start:start + window_size]
            feats = _extract_clip_features(window_frames, backbone, device, window_size, window_size)
            if not feats:
                continue
            label = 1 if window_mask.mean() > 0.3 else 0
            X_list.append(feats[0])
            y_list.append(label)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Dataset: {len(X)} samples | normal={int((y==0).sum())} | anomaly={int((y==1).sum())}")
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backbone = _build_backbone(device)

    X, y = build_dataset(
        train_videos_dir=Path(args.train_videos),
        test_frames_dir=Path(args.test_frames),
        test_masks_dir=Path(args.test_masks),
        backbone=backbone,
        device=device,
        window_size=args.window_size,
        stride=max(1, args.window_size // 4),
    )

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)

    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # Class weights to handle imbalance
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model     = WindowMLP(in_dim=_AGG_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(
                    F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight).item()
                )

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs} | train={t_loss:.4f} | val={v_loss:.4f} | lr={lr_now:.2e}")
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> Saved (val_loss={v_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-videos", default=str(_DEFAULT_TRAIN))
    parser.add_argument("--test-frames",  default=str(_DEFAULT_FRAMES))
    parser.add_argument("--test-masks",   default=str(_DEFAULT_MASKS))
    parser.add_argument("--window-size",  type=int,   default=30)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--out-path",     default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
