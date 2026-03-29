# CNN Autoencoder Anomaly Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Random Forest anomaly model with a CNN Autoencoder trained on normal videos, producing a reconstruction-error-based anomaly score that plugs into the existing `RollingInferencePipeline`.

**Architecture:** A Conv3D encoder-decoder autoencoder (input: 16-frame clip at 64×64) is trained exclusively on normal ShanghaiTech training videos using MSE reconstruction loss. At inference time, anomaly score = mean per-pixel reconstruction error (normalized), which is high for unseen abnormal patterns. This wraps into the same `anomaly_model_fn: (np.ndarray) -> float` interface already used by the pipeline.

**Tech Stack:** PyTorch, OpenCV, NumPy, scikit-learn (for AUC eval), existing `src/data/preprocess.py`, `src/inference/pipeline.py`

---

### Task 1: CNN Autoencoder Model

**Files:**
- Create: `src/models/autoencoder.py`
- Create: `tests/models/test_autoencoder_shapes.py`

**Step 1: Write the failing test**

```python
# tests/models/test_autoencoder_shapes.py
from __future__ import annotations
import torch
from src.models.autoencoder import Conv3DAutoencoder


def test_output_shape_matches_input():
    model = Conv3DAutoencoder()
    x = torch.randn(2, 3, 16, 64, 64)  # (B, C, T, H, W)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_latent_is_smaller_than_input():
    model = Conv3DAutoencoder()
    x = torch.randn(1, 3, 16, 64, 64)
    latent = model.encode(x)
    assert latent.numel() < x.numel()


def test_reconstruction_loss_is_scalar():
    import torch.nn.functional as F
    model = Conv3DAutoencoder()
    x = torch.randn(2, 3, 16, 64, 64)
    out = model(x)
    loss = F.mse_loss(out, x)
    assert loss.shape == torch.Size([])
```

**Step 2: Run test to verify it fails**

```bash
cd "D:\Traffic Camera Feed Project (Client 1)"
python -m pytest tests/models/test_autoencoder_shapes.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write the implementation**

```python
# src/models/autoencoder.py
from __future__ import annotations

import torch
from torch import nn


class Conv3DAutoencoder(nn.Module):
    """Conv3D autoencoder for video anomaly detection.

    Trained on normal-only clips. High reconstruction error => anomaly.
    Input shape: (B, C=3, T=16, H=64, W=64)
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Encoder: downsample T by 4, H/W by 8
        self.encoder = nn.Sequential(
            # (B, 3, 16, 64, 64) -> (B, c, 16, 32, 32)
            nn.Conv3d(3, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            # -> (B, c*2, 8, 16, 16)
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # -> (B, c*4, 4, 8, 8)
            nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        # Decoder: upsample back to original
        self.decoder = nn.Sequential(
            # (B, c*4, 4, 8, 8) -> (B, c*2, 8, 16, 16)
            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            # -> (B, c, 16, 32, 32)
            nn.ConvTranspose3d(c * 2, c, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            # -> (B, 3, 16, 64, 64)
            nn.ConvTranspose3d(c, 3, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/models/test_autoencoder_shapes.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/models/autoencoder.py tests/models/test_autoencoder_shapes.py
git commit -m "feat: add Conv3DAutoencoder model with encoder-decoder architecture"
```

---

### Task 2: Dataset Loader for Normal Training Videos

**Files:**
- Create: `src/data/normal_clip_dataset.py`
- Create: `tests/data/test_normal_clip_dataset.py`

**Step 1: Write the failing test**

```python
# tests/data/test_normal_clip_dataset.py
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path
from src.data.normal_clip_dataset import NormalClipDataset


def test_dataset_returns_tensor_shape(tmp_path):
    """Uses a synthetic video (random frames written as jpg) to test the loader."""
    import cv2
    clip_dir = tmp_path / "vid1"
    clip_dir.mkdir()
    for i in range(20):
        frame = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(clip_dir / f"{i:04d}.jpg"), frame)

    dataset = NormalClipDataset(
        video_dirs=[clip_dir],
        clip_len=16,
        frame_size=(64, 64),
        stride=8,
    )
    assert len(dataset) >= 1
    clip = dataset[0]
    # shape: (C=3, T=16, H=64, W=64), float32 in [0, 1]
    assert clip.shape == (3, 16, 64, 64)
    assert clip.dtype == np.float32
    assert clip.min() >= 0.0 and clip.max() <= 1.0


def test_dataset_empty_when_no_dirs():
    dataset = NormalClipDataset(video_dirs=[], clip_len=16, frame_size=(64, 64), stride=8)
    assert len(dataset) == 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/data/test_normal_clip_dataset.py -v
```

Expected: FAIL with `ImportError`

**Step 3: Write the implementation**

```python
# src/data/normal_clip_dataset.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class NormalClipDataset(Dataset):
    """Loads overlapping fixed-length clips from directories of JPEG frames.

    Each directory represents one video (frames named numerically).
    Used for training the CNN autoencoder on normal-only footage.

    Parameters
    ----------
    video_dirs : list of directories, each containing numbered .jpg frames
    clip_len   : number of frames per clip (T)
    frame_size : (width, height) to resize each frame to
    stride     : step between clip start positions (overlap = clip_len - stride)
    """

    def __init__(
        self,
        video_dirs: list[Path],
        clip_len: int,
        frame_size: tuple[int, int],
        stride: int,
    ) -> None:
        self._clip_len = clip_len
        self._frame_size = frame_size
        self._clips: list[tuple[list[Path], int]] = []  # (frame_paths, start_idx)

        for d in video_dirs:
            frames = sorted(Path(d).glob("*.jpg"), key=lambda p: int(p.stem))
            if len(frames) < clip_len:
                continue
            for start in range(0, len(frames) - clip_len + 1, stride):
                self._clips.append((frames, start))

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> np.ndarray:
        frames_list, start = self._clips[idx]
        clip = []
        w, h = self._frame_size
        for frame_path in frames_list[start : start + self._clip_len]:
            img = cv2.imread(str(frame_path))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip.append(img.astype(np.float32) / 255.0)
        # Stack to (T, H, W, C) then transpose to (C, T, H, W)
        arr = np.stack(clip, axis=0)          # (T, H, W, 3)
        return arr.transpose(3, 0, 1, 2)      # (3, T, H, W)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/data/test_normal_clip_dataset.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/data/normal_clip_dataset.py tests/data/test_normal_clip_dataset.py
git commit -m "feat: add NormalClipDataset for loading frame dirs as video clips"
```

---

### Task 3: Training Script

**Files:**
- Create: `scripts/train_autoencoder.py`

No unit test for script entrypoints (already covered by `tests/scripts/test_script_entrypoints.py` pattern). This is a runnable script.

**Step 1: Write the script**

```python
# scripts/train_autoencoder.py
"""Train a Conv3D autoencoder on ShanghaiTech normal training videos.

Usage
-----
    python scripts/train_autoencoder.py [--epochs 20] [--batch-size 4] [--lr 1e-3]
                                        [--data-dir data/raw/shanghaitech/shanghaitech/training/videos]
                                        [--out-path artifacts/models/autoencoder.pt]

The script saves the model state dict to --out-path on completion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.normal_clip_dataset import NormalClipDataset
from src.models.autoencoder import Conv3DAutoencoder

_DEFAULT_DATA = Path("data/raw/shanghaitech/shanghaitech/training/videos")
_DEFAULT_OUT = Path("artifacts/models/autoencoder.pt")

CLIP_LEN = 16
FRAME_SIZE = (64, 64)
STRIDE = 8


def collate_fn(batch: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(batch, axis=0))


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(data_dir.iterdir()) if data_dir.exists() else []
    if not video_dirs:
        raise FileNotFoundError(f"No video directories found in {data_dir}")

    print(f"Found {len(video_dirs)} video dirs")

    dataset = NormalClipDataset(
        video_dirs=video_dirs,
        clip_len=CLIP_LEN,
        frame_size=FRAME_SIZE,
        stride=STRIDE,
    )
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
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
            for batch in val_loader:
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

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conv3D autoencoder on normal videos")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--out-path", type=str, default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs (dry check)**

```bash
cd "D:\Traffic Camera Feed Project (Client 1)"
python -c "import scripts.train_autoencoder"
```

Expected: no ImportError

**Step 3: Commit**

```bash
git add scripts/train_autoencoder.py
git commit -m "feat: add autoencoder training script with best-model checkpoint saving"
```

---

### Task 4: CNN Inference Adapter

**Files:**
- Create: `src/inference/cnn_anomaly_model.py`
- Create: `tests/inference/test_cnn_anomaly_model.py`

**Step 1: Write the failing test**

```python
# tests/inference/test_cnn_anomaly_model.py
from __future__ import annotations
import numpy as np
import pytest
import torch
from pathlib import Path


def _make_dummy_model(tmp_path: Path) -> Path:
    """Save a freshly initialised (untrained) autoencoder to a temp file."""
    from src.models.autoencoder import Conv3DAutoencoder
    model = Conv3DAutoencoder()
    out = tmp_path / "dummy_autoencoder.pt"
    torch.save(model.state_dict(), out)
    return out


def test_returns_float_in_unit_interval(tmp_path):
    model_path = _make_dummy_model(tmp_path)
    from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn
    fn = load_cnn_anomaly_model_fn(model_path=model_path)
    # Synthetic clip: 30 frames, 240x320 BGR uint8 (same as RF model input)
    clip = np.random.randint(0, 255, (30, 240, 320, 3), dtype=np.uint8)
    score = fn(clip)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_missing_model_raises(tmp_path):
    from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn
    with pytest.raises(FileNotFoundError):
        load_cnn_anomaly_model_fn(model_path=tmp_path / "nonexistent.pt")
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/inference/test_cnn_anomaly_model.py -v
```

Expected: FAIL with `ImportError`

**Step 3: Write the implementation**

```python
# src/inference/cnn_anomaly_model.py
"""CNN Autoencoder-based anomaly model adapter.

Wraps the trained Conv3DAutoencoder so it satisfies the same
``anomaly_model_fn: (np.ndarray) -> float`` interface used by
RollingInferencePipeline — a drop-in replacement for the RF model.

Usage
-----
    from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn

    anomaly_fn = load_cnn_anomaly_model_fn()
    pipeline = RollingInferencePipeline(
        camera_id="cam_01",
        clip_length=30,
        clip_stride=10,
        anomaly_model_fn=anomaly_fn,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.autoencoder import Conv3DAutoencoder

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "autoencoder.pt"

# Preprocessing constants (must match training)
_CLIP_LEN = 16
_FRAME_SIZE = (64, 64)   # (width, height)

# Reconstruction error normalization: typical trained MSE range [0, ~0.05]
# Scores above this cap are clipped to 1.0
_MSE_CAP = 0.05


def _clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
    """Convert raw clip (N, H, W, 3) BGR uint8 to model input tensor (1, 3, T, 64, 64)."""
    n = len(clip)
    # Sub-sample to _CLIP_LEN frames evenly
    indices = np.linspace(0, n - 1, _CLIP_LEN, dtype=int)
    frames = []
    w, h = _FRAME_SIZE
    for i in indices:
        frame = clip[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)

    arr = np.stack(frames, axis=0)          # (T, H, W, 3)
    arr = arr.transpose(3, 0, 1, 2)         # (3, T, H, W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, T, H, W)


def load_cnn_anomaly_model_fn(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> Callable[[np.ndarray], float]:
    """Load trained autoencoder and return an anomaly scoring callable.

    Parameters
    ----------
    model_path : path to a .pt state-dict file, or None to use the default artifact.
    device     : 'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    A function ``(clip: np.ndarray) -> float`` where:
        - ``clip`` is shape (N, H, W, 3) uint8 BGR frames (N >= 1).
        - return value is anomaly probability in [0, 1].
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"CNN autoencoder not found at {path}.\n"
            "Run scripts/train_autoencoder.py first."
        )

    _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Conv3DAutoencoder()
    model.load_state_dict(torch.load(path, map_location=_device))
    model.to(_device)
    model.eval()

    def anomaly_model_fn(clip: np.ndarray) -> float:
        with torch.no_grad():
            x = _clip_to_tensor(clip).to(_device)
            recon = model(x)
            mse = float(F.mse_loss(recon, x).item())
        # Normalize to [0, 1] using the expected MSE cap
        score = min(1.0, mse / _MSE_CAP)
        return float(np.clip(score, 0.0, 1.0))

    return anomaly_model_fn
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/inference/test_cnn_anomaly_model.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/inference/cnn_anomaly_model.py tests/inference/test_cnn_anomaly_model.py
git commit -m "feat: add CNN autoencoder inference adapter matching anomaly_model_fn interface"
```

---

### Task 5: Evaluation Script

**Files:**
- Create: `scripts/evaluate_autoencoder.py`

**Step 1: Write the script**

```python
# scripts/evaluate_autoencoder.py
"""Evaluate the trained CNN autoencoder on the ShanghaiTech test set.

Computes frame-level ROC-AUC and PR-AUC to compare against the RF baseline.

Usage
-----
    python scripts/evaluate_autoencoder.py \
        [--frames-dir data/raw/shanghaitech/shanghaitech/testing/frames] \
        [--masks-dir data/raw/shanghaitech/shanghaitech/testing/test_frame_mask] \
        [--model-path artifacts/models/autoencoder.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.inference.cnn_anomaly_model import load_cnn_anomaly_model_fn

_DEFAULT_FRAMES = Path("data/raw/shanghaitech/shanghaitech/testing/frames")
_DEFAULT_MASKS  = Path("data/raw/shanghaitech/shanghaitech/testing/test_frame_mask")
_DEFAULT_MODEL  = Path("artifacts/models/autoencoder.pt")

# Frames per clip sent to anomaly_model_fn (matches pipeline clip_length)
_CLIP_LEN_PIPELINE = 30


def _load_clip_frames(frame_dir: Path) -> np.ndarray:
    """Load all jpg frames from a clip directory as (N, H, W, 3) uint8 BGR."""
    import cv2
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = [cv2.imread(str(p)) for p in paths]
    return np.stack(frames, axis=0)


def evaluate(args: argparse.Namespace) -> None:
    frames_root = Path(args.frames_dir)
    masks_root  = Path(args.masks_dir)
    model_path  = Path(args.model_path)

    anomaly_fn = load_cnn_anomaly_model_fn(model_path=model_path)

    clip_dirs = sorted(frames_root.iterdir())
    all_scores: list[float] = []
    all_labels: list[int] = []

    for clip_dir in clip_dirs:
        mask_file = masks_root / f"{clip_dir.name}.npy"
        if not mask_file.exists():
            print(f"  [skip] no mask for {clip_dir.name}")
            continue

        frames = _load_clip_frames(clip_dir)        # (N, H, W, 3)
        mask   = np.load(str(mask_file))            # (N,) binary

        n = len(frames)
        # Score each window of _CLIP_LEN_PIPELINE frames, assign score to centre frame
        half = _CLIP_LEN_PIPELINE // 2
        frame_scores = np.zeros(n, dtype=np.float32)
        counts       = np.zeros(n, dtype=np.int32)

        for start in range(0, n - _CLIP_LEN_PIPELINE + 1, max(1, _CLIP_LEN_PIPELINE // 4)):
            end = start + _CLIP_LEN_PIPELINE
            clip = frames[start:end]
            score = anomaly_fn(clip)
            centre = start + half
            frame_scores[centre] += score
            counts[centre] += 1

        # Fill un-scored frames with nearest scored value
        valid = counts > 0
        frame_scores[valid] /= counts[valid]
        # Simple forward-fill for gaps
        last = 0.0
        for i in range(n):
            if counts[i] > 0:
                last = frame_scores[i]
            else:
                frame_scores[i] = last

        all_scores.extend(frame_scores.tolist())
        all_labels.extend(mask.astype(int).tolist())
        print(f"  {clip_dir.name}: {n} frames, anomaly%={mask.mean()*100:.1f}%")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_scores)
    pr_auc  = average_precision_score(all_labels, all_scores)

    print(f"\n=== CNN Autoencoder Results ===")
    print(f"Clips evaluated : {len(clip_dirs)}")
    print(f"Total frames    : {len(all_labels)}")
    print(f"ROC-AUC         : {roc_auc:.4f}  (RF baseline: 0.8313)")
    print(f"PR-AUC          : {pr_auc:.4f}  (RF baseline: 0.8261)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN autoencoder on ShanghaiTech test set")
    parser.add_argument("--frames-dir", default=str(_DEFAULT_FRAMES))
    parser.add_argument("--masks-dir",  default=str(_DEFAULT_MASKS))
    parser.add_argument("--model-path", default=str(_DEFAULT_MODEL))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
```

**Step 2: Verify script is importable**

```bash
python -c "import scripts.evaluate_autoencoder"
```

Expected: no error

**Step 3: Commit**

```bash
git add scripts/evaluate_autoencoder.py
git commit -m "feat: add autoencoder evaluation script with ROC-AUC vs RF baseline comparison"
```

---

### Task 6: Run Full Test Suite

**Step 1: Run all new tests**

```bash
python -m pytest tests/models/test_autoencoder_shapes.py tests/data/test_normal_clip_dataset.py tests/inference/test_cnn_anomaly_model.py -v
```

Expected: All PASS

**Step 2: Run full suite to check no regressions**

```bash
python -m pytest --tb=short -q
```

Expected: All existing tests still pass

**Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address any regression from CNN autoencoder integration"
```

---

### Task 7: Train the Model

> Do this interactively — it will take 10-30 minutes depending on hardware.

**Step 1: Run training**

```bash
python scripts/train_autoencoder.py --epochs 20 --batch-size 4
```

Watch for: decreasing `val_mse` each epoch. Model saved to `artifacts/models/autoencoder.pt`.

**Step 2: Run evaluation vs RF baseline**

```bash
python scripts/evaluate_autoencoder.py
```

Expected output shows `ROC-AUC` and compares to RF baseline 0.8313.

**Step 3: Commit artifact (if small enough) or add to .gitignore**

```bash
# If model file < 100MB:
git add artifacts/models/autoencoder.pt
git commit -m "feat: add trained CNN autoencoder artifact (20 epochs)"
```

---

## Summary

| Task | What it adds |
|------|-------------|
| 1    | `Conv3DAutoencoder` model class |
| 2    | `NormalClipDataset` data loader |
| 3    | `train_autoencoder.py` training script |
| 4    | `cnn_anomaly_model.py` inference adapter |
| 5    | `evaluate_autoencoder.py` evaluation vs RF |
| 6    | Full test suite verification |
| 7    | Actual training + evaluation run |

The CNN model plugs into the existing pipeline with zero changes to `RollingInferencePipeline` or the dashboard.
