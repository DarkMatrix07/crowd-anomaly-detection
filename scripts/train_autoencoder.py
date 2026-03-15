# scripts/train_autoencoder.py
"""Train a Conv3D autoencoder on ShanghaiTech normal training videos.

Usage
-----
    python scripts/train_autoencoder.py [--epochs 50] [--batch-size 2] [--lr 1e-3]
                                        [--data-dir data/raw/shanghaitech/shanghaitech/training/videos]
                                        [--out-path artifacts/models/autoencoder.pt]
                                        [--resume artifacts/models/autoencoder_checkpoint.pt]

The data-dir may contain either:
  - .avi video files  (ShanghaiTech training set layout)
  - sub-directories of JPEG frames  (pre-extracted layout)

Checkpoints
-----------
Every epoch a full checkpoint (model + optimizer + scheduler + epoch) is saved to
<out-path stem>_checkpoint.pt  (e.g. autoencoder_checkpoint.pt).
The best model weights-only file is still written to --out-path.

To resume training on another machine:
    python scripts/train_autoencoder.py --epochs 50 --resume artifacts/models/autoencoder_checkpoint.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.data.normal_clip_dataset import NormalClipDataset
from src.models.autoencoder import Conv3DAutoencoder

_DEFAULT_DATA = Path("data/raw/shanghaitech/shanghaitech/training/videos")
_DEFAULT_OUT = Path("artifacts/models/autoencoder.pt")

CLIP_LEN = 16
FRAME_SIZE = (128, 128)
STRIDE = 64


class _AviClipDataset(Dataset):
    """Extract overlapping clips directly from .avi files via OpenCV VideoCapture."""

    def __init__(self, video_paths: list[Path], clip_len: int,
                 frame_size: tuple[int, int], stride: int) -> None:
        self._clip_len = clip_len
        self._frame_size = frame_size
        # (video_path, start_frame_index)
        self._clips: list[tuple[Path, int]] = []

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n < clip_len:
                continue
            for start in range(0, n - clip_len + 1, stride):
                self._clips.append((vp, start))

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> np.ndarray:
        vp, start = self._clips[idx]
        cap = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        w, h = self._frame_size
        frames = []
        for _ in range(self._clip_len):
            try:
                ok, frame = cap.read()
            except cv2.error:
                ok = False
            if not ok or frame is None:
                # Pad with last good frame or zeros
                frames.append(frames[-1] if frames else np.zeros((h, w, 3), dtype=np.float32))
                continue
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        arr = np.stack(frames, axis=0)       # (T, H, W, 3)
        return arr.transpose(3, 0, 1, 2)     # (3, T, H, W)


def _build_dataset(data_dir: Path) -> Dataset:
    """Return the right dataset depending on whether data_dir has .avi files or frame dirs."""
    entries = sorted(data_dir.iterdir())
    if not entries:
        raise FileNotFoundError(f"No files found in {data_dir}")

    if entries[0].suffix.lower() == ".avi":
        print(f"Detected .avi files — using VideoCapture loader ({len(entries)} videos)")
        return _AviClipDataset(entries, clip_len=CLIP_LEN, frame_size=FRAME_SIZE, stride=STRIDE)
    else:
        print(f"Detected frame directories — using JPEG loader ({len(entries)} dirs)")
        return NormalClipDataset(entries, clip_len=CLIP_LEN, frame_size=FRAME_SIZE, stride=STRIDE)


def collate_fn(batch: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(batch, axis=0))


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.with_name(out_path.stem + "_checkpoint.pt")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dataset = _build_dataset(data_dir)
    print(f"Total clips: {len(dataset)}")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = Conv3DAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint if requested
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (best val MSE so far: {best_val_loss:.6f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs} [train]", leave=False):
            batch = batch.to(device)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{args.epochs} [val]", leave=False):
                batch = batch.to(device)
                recon = model(batch)
                val_losses.append(F.mse_loss(recon, batch).item())

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_mse={t_loss:.6f} | val_mse={v_loss:.6f}")
        scheduler.step()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> Saved best model (val_mse={v_loss:.6f})")

        # Save full checkpoint every epoch for resumability
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, checkpoint_path)

        # Free memory between epochs to prevent OOM over long runs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conv3D autoencoder on normal videos")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--out-path", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume training from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
