"""Demo endpoints — serve frames and run model inference for the React dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(prefix="/demo", tags=["demo"])

FRAMES_ROOT = PROJECT_ROOT / "data/demo_clips/frames"
MASKS_ROOT  = PROJECT_ROOT / "data/demo_clips/masks"

DEMO_CLIPS = {
    "01_0130": "Scene 01 — Sudden crowd rush (peaks 0.997)",
    "02_0128": "Scene 02 — Dense crowd anomaly (55% anomaly)",
    "01_0063": "Scene 01 — Fast-escalating aggression",
    "01_0054": "Scene 01 — Early warning, slow build-up",
    "05_0018": "Scene 05 — Normal pedestrian flow (baseline)",
    "03_0031": "Scene 03 — Loitering and irregular movement",
    "04_0001": "Scene 04 — Chaotic crowd scatter (52% anomaly)",
    "06_0144": "Scene 06 — High-density anomaly (58% anomaly)",
    "07_0005": "Scene 07 — Sporadic abnormal behaviour",
    "08_0044": "Scene 08 — Running and chasing incident",
}


# ── List clips ────────────────────────────────────────────────────────────────

@router.get("/clips")
def list_clips() -> dict:
    clips = []
    for clip_id, description in DEMO_CLIPS.items():
        clip_dir = FRAMES_ROOT / clip_id
        if not clip_dir.exists():
            continue
        paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        mask_file = MASKS_ROOT / f"{clip_id}.npy"
        clips.append({
            "id":               clip_id,
            "description":      description,
            "frame_count":      len(paths),
            "has_ground_truth": mask_file.exists(),
        })
    if not clips:
        raise HTTPException(
            503,
            "Demo frames not found. Ensure data/demo_clips/frames/ exists in the repo.",
        )
    return {"clips": clips}


# ── Serve a single frame ──────────────────────────────────────────────────────

@router.get("/clips/{clip_id}/frame/{frame_idx}")
def get_frame(clip_id: str, frame_idx: int) -> FileResponse:
    clip_dir = FRAMES_ROOT / clip_id
    if not clip_dir.exists():
        raise HTTPException(404, f"Clip '{clip_id}' not found")
    paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if frame_idx < 0 or frame_idx >= len(paths):
        raise HTTPException(404, f"Frame {frame_idx} out of range (clip has {len(paths)} frames)")
    return FileResponse(str(paths[frame_idx]), media_type="image/jpeg")


# ── Run inference ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    model: str = "resnet"   # "resnet" | "rf"
    window_size: int = 30
    stride: int = 10


@router.post("/clips/{clip_id}/analyze")
def analyze_clip(clip_id: str, body: AnalyzeRequest) -> dict:
    clip_dir  = FRAMES_ROOT / clip_id
    mask_file = MASKS_ROOT / f"{clip_id}.npy"

    if not clip_dir.exists():
        raise HTTPException(404, f"Clip '{clip_id}' not found")

    # Load frames
    paths  = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    n = len(frames)
    if n == 0:
        raise HTTPException(400, "No readable frames found")

    frames_arr = np.stack(frames)

    # Load model
    if body.model == "resnet":
        from src.inference.resnet_mlp_model import load_resnet_mlp_model_fn
        anomaly_fn = load_resnet_mlp_model_fn()
    else:
        from src.inference.anomaly_model import load_anomaly_model_fn
        anomaly_fn = load_anomaly_model_fn()

    # Pre-score all windows
    frame_scores = np.zeros(n, dtype=np.float32)
    counts       = np.zeros(n, dtype=np.int32)
    half         = body.window_size // 2

    for start in range(0, n - body.window_size + 1, max(1, body.stride)):
        clip_window = frames_arr[start : start + body.window_size]
        score       = float(anomaly_fn(clip_window))
        centre      = start + half
        frame_scores[centre] += score
        counts[centre]       += 1

    # Fill gaps
    valid = counts > 0
    frame_scores[valid] /= counts[valid]
    last = 0.0
    for i in range(n):
        if counts[i] > 0:
            last = frame_scores[i]
        else:
            frame_scores[i] = last

    # Ground truth
    gt_labels: list[int] | None = None
    if mask_file.exists():
        gt_labels = np.load(str(mask_file)).astype(int).tolist()

    scores_list = [round(float(s), 4) for s in frame_scores]
    preds       = [int(s >= 0.5) for s in frame_scores]

    result: dict = {
        "clip_id":       clip_id,
        "model":         body.model,
        "frame_count":   n,
        "scores":        scores_list,
        "predictions":   preds,
        "gt_labels":     gt_labels,
        "peak_score":    round(float(frame_scores.max()), 4),
        "mean_score":    round(float(frame_scores.mean()), 4),
        "anomaly_frames": int((frame_scores >= 0.5).sum()),
    }

    if gt_labels is not None:
        gt_arr   = np.array(gt_labels)
        pred_arr = np.array(preds)
        result["accuracy"]        = round(float((pred_arr == gt_arr).mean()), 4)
        result["gt_anomaly_pct"]  = round(float(gt_arr.mean() * 100), 1)

        try:
            from sklearn.metrics import roc_auc_score
            result["roc_auc"] = round(float(roc_auc_score(gt_arr, frame_scores)), 4)
        except Exception:
            pass

    return result
