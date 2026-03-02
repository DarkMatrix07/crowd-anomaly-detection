"""End-to-end live demo script for client / supervisor presentation.

Simulates a camera feed by replaying a ShanghaiTech test clip frame-by-frame
through the full RollingInferencePipeline (real trained model).  Every alert
generated is posted to the running FastAPI server so it appears on the
Streamlit dashboard in real time.

Usage
-----
Step 1 – start the API (in a separate terminal):
    uvicorn src.api.app:app --reload --port 8000

Step 2 – start the dashboard (in another terminal):
    streamlit run dashboard/app.py

Step 3 – run this script:
    python scripts/demo_live.py --clip-id 01_0054
    python scripts/demo_live.py --clip-id 04_0004   # best early-warning demo
    python scripts/demo_live.py --clip-id 01_0130   # highest anomaly score

Press Ctrl+C to stop at any time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import _sorted_frame_paths
from src.inference.anomaly_model import load_anomaly_model_fn
from src.inference.pipeline import RollingInferencePipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live demo: replay a clip through the full pipeline")
    p.add_argument("--clip-id", default="04_0004",
                   help="ShanghaiTech testing clip ID (e.g. 01_0054, 04_0004, 01_0130)")
    p.add_argument("--dataset-root", default="data/raw/shanghaitech/shanghaitech")
    p.add_argument("--api-url", default="http://127.0.0.1:8000",
                   help="Base URL of the running FastAPI server")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Replay speed in frames-per-second (lower = slower, easier to watch)")
    p.add_argument("--clip-length", type=int, default=30,
                   help="Frames per inference window (must match training)")
    p.add_argument("--clip-stride", type=int, default=10,
                   help="Emit an inference result every N frames")
    p.add_argument("--no-api", action="store_true",
                   help="Run without posting to API (offline mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def check_api(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def post_alert(base_url: str, event: dict) -> None:
    payload = {
        "timestamp": event["timestamp"],
        "camera_id": event["camera_id"],
        "risk_level": event["risk_level"],
        "score": event["score"],
        "evidence_window": event["evidence_window"],
    }
    try:
        requests.post(f"{base_url}/alerts", json=payload, timeout=3)
    except Exception as e:
        print(f"  [warn] Could not post alert to API: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RISK_COLOURS = {
    "LOW":    "\033[92m",   # green
    "MEDIUM": "\033[93m",   # yellow
    "HIGH":   "\033[91m",   # red
}
RESET = "\033[0m"

def _colour(level: str, text: str) -> str:
    return f"{RISK_COLOURS.get(level, '')}{text}{RESET}"


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    clip_dir = dataset_root / "testing" / "frames" / args.clip_id
    mask_path = dataset_root / "testing" / "test_frame_mask" / f"{args.clip_id}.npy"

    if not clip_dir.exists():
        print(f"ERROR: Clip not found: {clip_dir}")
        return 1

    # Ground-truth mask (for display only — model doesn't see this)
    gt_mask = None
    if mask_path.exists():
        gt_mask = np.load(mask_path).astype(np.uint8).reshape(-1)

    # API check
    use_api = not args.no_api
    if use_api:
        if check_api(args.api_url):
            print(f"API connected at {args.api_url}")
        else:
            print(f"WARNING: API not reachable at {args.api_url}")
            print("  Start the API with: uvicorn src.api.app:app --reload --port 8000")
            print("  Continuing in offline mode (alerts will not be saved).\n")
            use_api = False

    # Load model
    print("Loading anomaly model ...")
    anomaly_fn = load_anomaly_model_fn()
    print("Model loaded.\n")

    # Build pipeline
    pipe = RollingInferencePipeline(
        camera_id=args.clip_id,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        anomaly_model_fn=anomaly_fn,
    )

    frame_paths = _sorted_frame_paths(clip_dir)
    total_frames = len(frame_paths)
    frame_delay = 1.0 / args.fps

    print(f"{'='*60}")
    print(f"  Clip     : {args.clip_id}")
    print(f"  Frames   : {total_frames}")
    print(f"  Replay   : {args.fps} fps  (real time = {total_frames/args.fps:.1f}s)")
    print(f"  Window   : {args.clip_length} frames  stride {args.clip_stride}")
    print(f"{'='*60}\n")

    alert_count = 0
    start_time = time.time()

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        timestamp = frame_idx / 25.0  # ShanghaiTech is ~25fps

        events = pipe.process_frame(frame, timestamp=timestamp)

        for event in events:
            alert_count += 1
            level = event["risk_level"]
            score = event["score"]
            gt = ""
            if gt_mask is not None and frame_idx < len(gt_mask):
                gt = "  [GT: ANOMALY]" if gt_mask[frame_idx] == 1 else "  [GT: normal]"

            print(
                f"  Frame {frame_idx:4d} | t={timestamp:.2f}s | "
                f"Score={score:.3f} | {_colour(level, level)}{gt}"
            )

            if use_api:
                post_alert(args.api_url, event)

        # Pace the replay
        elapsed = time.time() - start_time
        expected = (frame_idx + 1) * frame_delay
        if expected > elapsed:
            time.sleep(expected - elapsed)

    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Done. {total_frames} frames in {duration:.1f}s")
    print(f"  Total alerts generated : {alert_count}")
    if use_api:
        print(f"  Alerts posted to API   : {args.api_url}/alerts")
        print(f"  Open dashboard         : streamlit run dashboard/app.py")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
