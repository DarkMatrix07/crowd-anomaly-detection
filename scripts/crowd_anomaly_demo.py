"""Visual live demo — Streamlit app.

Shows actual video frames with live anomaly scoring overlaid.
Bypass RollingInferencePipeline to use raw model probabilities directly
(the pipeline's risk formula dilutes the score to ~40% of the model output,
keeping everything LOW — this demo shows the real model confidence).

Run with:
    python -m streamlit run scripts/demo_visual.py
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.anomaly_features import _sorted_frame_paths, compute_frame_features

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crowd Anomaly Monitor",
    page_icon="🚨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "shanghaitech" / "shanghaitech"
FRAMES_ROOT  = DATASET_ROOT / "testing" / "frames"
MASKS_ROOT   = DATASET_ROOT / "testing" / "test_frame_mask"
MODEL_PATH   = PROJECT_ROOT / "artifacts" / "models" / "shanghaitech_windowed_rf.joblib"

DEMO_CLIPS = {
    "01_0130  —  score peaks at 0.997  ★ best demo":     "01_0130",
    "02_0128  —  score peaks at 0.960":                   "02_0128",
    "01_0063  —  fast-escalating anomaly":                "01_0063",
    "01_0054  —  12-frame early warning":                 "01_0054",
    "05_0018  —  normal clip  (should stay LOW)":         "05_0018",
}

# Thresholds for display — match training evaluation
THRESH_MEDIUM = 0.50
THRESH_HIGH   = 0.75

LEVEL_COLOR = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
LEVEL_ICON  = {"LOW": "🟢",      "MEDIUM": "🟡",       "HIGH": "🔴"}

WINDOW_SIZE   = 30
WINDOW_STRIDE = 10
RESIZE        = (320, 240)
FRAME_DIM     = 10


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading anomaly model…")
def get_model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Feature extraction (same logic as train_windowed.py)
# ---------------------------------------------------------------------------
def frames_to_window_feat(frames: list[np.ndarray]) -> np.ndarray:
    """Convert list of BGR frames → 40-d window feature vector (1, 40)."""
    feats = []
    previous = None
    for frame in frames:
        f = cv2.resize(frame, RESIZE, interpolation=cv2.INTER_AREA)
        feat = compute_frame_features(f, prev_frame=previous)
        previous = f
        feats.append(feat)

    arr   = np.vstack(feats).astype(np.float32)      # (W, 10)
    third = max(1, len(arr) // 3)
    vec   = np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.max(axis=0),
        arr[-third:].mean(axis=0) - arr[:third].mean(axis=0),
    ])
    return vec.reshape(1, -1)


def score_to_level(score: float) -> str:
    if score >= THRESH_HIGH:
        return "HIGH"
    if score >= THRESH_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Frame overlay
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    score: float,
    level: str,
    frame_idx: int,
    gt_label: int | None,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Semi-transparent banner
    banner = vis.copy()
    cv2.rectangle(banner, (0, 0), (w, 54), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.55, vis, 0.45, 0, vis)

    # Colour-coded score bar at bottom of banner
    bar_bgr = {
        "LOW":    (39, 174, 96),
        "MEDIUM": (243, 156, 18),
        "HIGH":   (231, 76,  60),
    }
    col = tuple(reversed(bar_bgr.get(level, (100, 100, 100))))
    cv2.rectangle(vis, (0, 46), (int(score * w), 54), col, -1)

    # Text
    cv2.putText(vis, f"Score: {score:.3f}   |   {level}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame {frame_idx}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Ground truth (top-right)
    if gt_label is not None:
        gt_text  = "GT: ANOMALY" if gt_label == 1 else "GT: normal"
        gt_color = (60, 60, 230) if gt_label == 1 else (40, 200, 40)
        cv2.putText(vis, gt_text, (w - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, gt_color, 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🚨 Crowd Monitor")
    st.markdown("---")

    clip_label = st.selectbox("Select test clip", list(DEMO_CLIPS.keys()))
    clip_id    = DEMO_CLIPS[clip_label]
    fps        = st.slider("Replay speed (fps)", 1, 25, value=8)
    show_gt    = st.checkbox("Show ground-truth overlay", value=True)

    st.markdown("---")
    st.markdown(
        f"**Thresholds**\n\n"
        f"🟢 LOW  < {THRESH_MEDIUM}\n\n"
        f"🟡 MEDIUM  {THRESH_MEDIUM}–{THRESH_HIGH}\n\n"
        f"🔴 HIGH  ≥ {THRESH_HIGH}"
    )
    st.markdown("---")
    st.caption(
        "**Model:** RandomForest · W=30 frames · 40-d features\n\n"
        "**Dataset:** ShanghaiTech Campus\n\n"
        "**ROC-AUC:** 0.831  ·  **Recall:** 88.5%"
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("Abnormal Crowd Behaviour Detection — Live Demo")
st.caption(f"Clip: `{clip_id}`  ·  Window: {WINDOW_SIZE} frames  ·  Stride: {WINDOW_STRIDE} frames")

col_vid, col_stats = st.columns([3, 2])

with col_vid:
    frame_ph = st.empty()

with col_stats:
    st.subheader("Live Anomaly Score")
    metric_ph = st.empty()
    level_ph  = st.empty()
    bar_ph    = st.empty()
    st.subheader("Score Over Time")
    chart_ph  = st.empty()
    st.subheader("Alert Log")
    log_ph    = st.empty()

start = st.button("▶  Start Demo", type="primary", use_container_width=True)

if not start:
    frame_ph.info("👆 Press **Start Demo** to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Load clip
# ---------------------------------------------------------------------------
clip_dir  = FRAMES_ROOT / clip_id
mask_path = MASKS_ROOT / f"{clip_id}.npy"

if not clip_dir.exists():
    st.error(f"Clip not found: {clip_dir}")
    st.stop()

frame_paths = _sorted_frame_paths(clip_dir)
gt_mask     = np.load(mask_path).astype(np.uint8).reshape(-1) if mask_path.exists() else None
model       = get_model()

# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------
frame_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
score_history: list[dict]       = []
alert_log: list[dict]           = []

current_score = 0.0
current_level = "LOW"
frame_delay   = 1.0 / fps

for frame_idx, frame_path in enumerate(frame_paths):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    frame_buffer.append(frame)
    gt_label = int(gt_mask[frame_idx]) if (gt_mask is not None and frame_idx < len(gt_mask)) else None

    # Score every WINDOW_STRIDE frames once buffer is full
    if len(frame_buffer) == WINDOW_SIZE and frame_idx % WINDOW_STRIDE == 0:
        window_feat     = frames_to_window_feat(list(frame_buffer))
        current_score   = float(model.predict_proba(window_feat)[0, 1])
        current_level   = score_to_level(current_score)
        score_history.append({"frame": frame_idx, "score": current_score})

        if current_level in ("MEDIUM", "HIGH"):
            alert_log.append({
                "frame":    frame_idx,
                "time (s)": f"{frame_idx / 25:.2f}",
                "level":    current_level,
                "score":    f"{current_score:.3f}",
                "gt":       "ANOMALY" if gt_label == 1 else "normal",
            })

    # --- Video frame ---
    vis = draw_overlay(frame, current_score, current_level, frame_idx,
                       gt_label if show_gt else None)
    frame_ph.image(vis, use_container_width=True)

    # --- Score metric ---
    metric_ph.metric("Anomaly Score", f"{current_score:.3f}")

    # --- Level badge ---
    color = LEVEL_COLOR[current_level]
    icon  = LEVEL_ICON[current_level]
    level_ph.markdown(
        f"<div style='background:{color};border-radius:8px;padding:14px;"
        f"text-align:center;font-size:1.5em;font-weight:bold;color:white'>"
        f"{icon} {current_level}</div>",
        unsafe_allow_html=True,
    )

    # --- Progress bar ---
    bar_ph.progress(min(current_score, 1.0))

    # --- Chart ---
    if len(score_history) > 1:
        df = pd.DataFrame(score_history).set_index("frame")
        chart_ph.line_chart(df, height=180)

    # --- Alert log ---
    if alert_log:
        log_ph.dataframe(
            pd.DataFrame(alert_log[::-1]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        log_ph.info("No MEDIUM / HIGH alerts yet.")

    time.sleep(frame_delay)

# Done
st.success(
    f"✅ Complete — {len(frame_paths)} frames, "
    f"{len(alert_log)} MEDIUM/HIGH alerts generated."
)
