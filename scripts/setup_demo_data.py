"""Setup demo data by copying the 5 demo clips from the full dataset.

Usage:
    python scripts/setup_demo_data.py

This script copies the 5 pre-selected demo clips from data/raw/ into
data/demo/, which is the folder the Streamlit demo app uses.

Run this once after cloning the repo (if you have the full ShanghaiTech
dataset placed in data/raw/).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FRAMES = PROJECT_ROOT / "data" / "raw" / "shanghaitech" / "shanghaitech" / "testing" / "frames"
SRC_MASKS  = PROJECT_ROOT / "data" / "raw" / "shanghaitech" / "shanghaitech" / "testing" / "test_frame_mask"
DST_FRAMES = PROJECT_ROOT / "data" / "demo" / "frames"
DST_MASKS  = PROJECT_ROOT / "data" / "demo" / "masks"

DEMO_CLIPS = ["01_0130", "02_0128", "01_0063", "01_0054", "05_0018"]

GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"


def info(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC}  {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC}  {msg}")


def error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}")


def main() -> int:
    print("=" * 60)
    print("  Demo Data Setup")
    print("=" * 60)

    if not SRC_FRAMES.exists():
        error("Full dataset not found.")
        error(f"Expected at: {SRC_FRAMES}")
        error("")
        error("Download the ShanghaiTech dataset and place it at:")
        error("  data/raw/shanghaitech/shanghaitech/")
        error("")
        error("Alternatively, use 'Upload Video' mode in the demo app")
        error("to analyse your own video files without the dataset.")
        return 1

    DST_FRAMES.mkdir(parents=True, exist_ok=True)
    DST_MASKS.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for clip_id in DEMO_CLIPS:
        src_clip = SRC_FRAMES / clip_id
        dst_clip = DST_FRAMES / clip_id

        if not src_clip.exists():
            warn(f"Clip not found in dataset: {clip_id} — skipping")
            all_ok = False
            continue

        if dst_clip.exists() and any(dst_clip.iterdir()):
            info(f"Already exists: {clip_id}")
            continue

        info(f"Copying frames: {clip_id} ...")
        shutil.copytree(src_clip, dst_clip, dirs_exist_ok=True)
        frame_count = sum(1 for _ in dst_clip.glob("*.jpg"))
        info(f"  -> {frame_count} frames copied")

        # Copy mask if available
        src_mask = SRC_MASKS / f"{clip_id}.npy"
        if src_mask.exists():
            shutil.copy2(src_mask, DST_MASKS / f"{clip_id}.npy")
            info(f"  -> mask copied")
        else:
            warn(f"  Mask not found for {clip_id}")

    print()
    if all_ok:
        info("Demo data setup complete. Run START_DEMO.sh to launch.")
    else:
        warn("Setup completed with warnings. Some clips may be missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
