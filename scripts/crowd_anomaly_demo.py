"""Visual live demo — Streamlit app.

Shows actual video frames with live anomaly scoring overlaid.
Pre-scores all frames before playback so the video runs smoothly
without inference lag during replay.

Run with:
    python -m streamlit run scripts/crowd_anomaly_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
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
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark professional theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}

/* Alert badge */
.alert-badge {
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    transition: background 0.3s ease;
}
.alert-LOW    { background: linear-gradient(135deg, #0d3321, #1a5c38); color: #3fb950; border: 1px solid #238636; }
.alert-MEDIUM { background: linear-gradient(135deg, #3d2a00, #6d4900); color: #f0883e; border: 1px solid #9e6a03; }
.alert-HIGH   { background: linear-gradient(135deg, #3d0000, #6d1010); color: #ff7b72; border: 1px solid #da3633;
                animation: pulse 1s ease-in-out infinite; }

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.4); }
    50%       { box-shadow: 0 0 0 10px rgba(218, 54, 51, 0); }
}

/* Score bar */
.score-bar-wrap {
    background: #21262d;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 12px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease, background 0.3s ease;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}

/* Alert table */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* Title */
h1 { color: #e6edf3 !important; font-weight: 700 !important; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Sidebar text */
[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
[data-testid="stSidebar"] label { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "shanghaitech" / "shanghaitech"
FRAMES_ROOT  = DATASET_ROOT / "testing" / "frames"
MASKS_ROOT   = DATASET_ROOT / "testing" / "test_frame_mask"
MODEL_PATH   = PROJECT_ROOT / "artifacts" / "models" / "shanghaitech_windowed_rf.joblib"

DEMO_CLIPS = {
    "01_0130  —  peaks at 0.997  ★ best demo":  "01_0130",
    "02_0128  —  peaks at 0.960":               "02_0128",
    "01_0063  —  fast-escalating anomaly":       "01_0063",
    "01_0054  —  12-frame early warning":        "01_0054",
    "05_0018  —  normal clip (stays LOW)":       "05_0018",
}

THRESH_MEDIUM = 0.50
THRESH_HIGH   = 0.75

LEVEL_ICON  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

BAR_COLOR = {
    "LOW":    "#3fb950",
    "MEDIUM": "#f0883e",
    "HIGH":   "#ff7b72",
}

WINDOW_SIZE   = 30
WINDOW_STRIDE = 10
RESIZE        = (320, 240)

# How often to refresh the chart during playback (every N frames)
CHART_REFRESH = 5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading anomaly model…")
def get_model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Video upload helpers
# ---------------------------------------------------------------------------
MAX_UPLOAD_FRAMES = 1500  # cap to avoid memory issues on very long videos


def extract_video_frames(video_bytes: bytes) -> list[np.ndarray]:
    """Write uploaded bytes to a temp file and extract all frames via OpenCV."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    frames: list[np.ndarray] = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        while len(frames) < MAX_UPLOAD_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def frames_to_window_feat(frames: list[np.ndarray]) -> np.ndarray:
    feats = []
    previous = None
    for frame in frames:
        f = cv2.resize(frame, RESIZE, interpolation=cv2.INTER_AREA)
        feat = compute_frame_features(f, prev_frame=previous)
        previous = f
        feats.append(feat)
    arr   = np.vstack(feats).astype(np.float32)
    third = max(1, len(arr) // 3)
    vec   = np.concatenate([
        arr.mean(axis=0), arr.std(axis=0),
        arr.max(axis=0),
        arr[-third:].mean(axis=0) - arr[:third].mean(axis=0),
    ])
    return vec.reshape(1, -1)


def score_to_level(score: float) -> str:
    if score >= THRESH_HIGH:   return "HIGH"
    if score >= THRESH_MEDIUM: return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Frame overlay (drawn with OpenCV, embedded in the video image itself)
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    score: float,
    level: str,
    frame_idx: int,
    total_frames: int,
    gt_label: int | None,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    bar_bgr = {"LOW": (40, 185, 63), "MEDIUM": (30, 136, 229), "HIGH": (60, 76, 231)}

    # Top banner
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (13, 17, 23), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    # Score text
    col_bgr = bar_bgr.get(level, (150, 150, 150))
    cv2.putText(vis, f"Score: {score:.3f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 237, 243), 2, cv2.LINE_AA)
    cv2.putText(vis, level, (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col_bgr, 2, cv2.LINE_AA)

    # Score bar (thin strip at bottom of banner)
    fill_w = int(score * w)
    cv2.rectangle(vis, (0, 52), (w, 60), (33, 38, 45), -1)
    cv2.rectangle(vis, (0, 52), (fill_w, 60), col_bgr, -1)

    # Frame counter (bottom-left)
    cv2.putText(vis, f"{frame_idx + 1} / {total_frames}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 110, 120), 1, cv2.LINE_AA)

    # Ground truth label (top-right corner, small)
    if gt_label is not None:
        gt_text  = "GT: ANOMALY" if gt_label == 1 else "GT: normal"
        gt_color = (80, 80, 220) if gt_label == 1 else (40, 185, 63)
        cv2.putText(vis, gt_text, (w - 200, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, gt_color, 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚨 Crowd Monitor")

    st.markdown("<div class='section-header'>Source</div>", unsafe_allow_html=True)
    input_mode = st.radio(
        "Input mode",
        ["Demo Clips", "Upload Video"],
        label_visibility="collapsed",
        horizontal=True,
    )

    # --- Demo Clips mode ---
    clip_id: str | None = None
    uploaded_frames: list[np.ndarray] | None = None
    upload_name: str = ""

    if input_mode == "Demo Clips":
        st.markdown("<div class='section-header'>Clip Selection</div>", unsafe_allow_html=True)
        clip_label = st.selectbox("Select clip", list(DEMO_CLIPS.keys()), label_visibility="collapsed")
        clip_id    = DEMO_CLIPS[clip_label]
    else:
        st.markdown("<div class='section-header'>Upload Video</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            upload_name = uploaded_file.name
            with st.spinner("Reading video…"):
                uploaded_frames = extract_video_frames(uploaded_file.read())
            st.success(f"{len(uploaded_frames)} frames loaded from **{upload_name}**")
            if len(uploaded_frames) == MAX_UPLOAD_FRAMES:
                st.warning(f"Capped at {MAX_UPLOAD_FRAMES} frames. Longer sections will be ignored.")
        else:
            st.info("Upload an MP4, AVI, MOV, or MKV file to analyse.")

    st.markdown("<div class='section-header'>Playback</div>", unsafe_allow_html=True)
    fps = st.slider("Speed (fps)", 1, 30, value=12)
    show_gt = (input_mode == "Demo Clips") and st.checkbox("Show ground-truth label", value=True)

    st.markdown("<div class='section-header'>Thresholds</div>", unsafe_allow_html=True)
    st.markdown(
        f"🟢 &nbsp;**LOW** — score < {THRESH_MEDIUM}<br>"
        f"🟡 &nbsp;**MEDIUM** — {THRESH_MEDIUM} to {THRESH_HIGH}<br>"
        f"🔴 &nbsp;**HIGH** — score ≥ {THRESH_HIGH}",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Model Info</div>", unsafe_allow_html=True)
    st.markdown(
        "**Type:** RandomForest · W=30 frames  \n"
        "**Features:** 40-d temporal window  \n"
        "**ROC-AUC:** 0.831 &nbsp;·&nbsp; **Recall:** 88.5%",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.markdown("# Abnormal Crowd Behaviour Detection")
_source_label = f"<code>{clip_id}</code>" if input_mode == "Demo Clips" else f"<code>{upload_name or 'no file'}</code>"
st.markdown(
    f"<span style='color:#8b949e;font-size:0.9rem'>Source: {_source_label} &nbsp;·&nbsp; "
    f"Window: {WINDOW_SIZE} frames &nbsp;·&nbsp; Stride: {WINDOW_STRIDE} frames</span>",
    unsafe_allow_html=True,
)

col_vid, col_stats = st.columns([3, 2], gap="large")

with col_vid:
    frame_ph = st.empty()

with col_stats:
    level_ph  = st.empty()
    score_ph  = st.empty()
    bar_ph    = st.empty()

    st.markdown("<div class='section-header'>Score Over Time</div>", unsafe_allow_html=True)
    chart_ph = st.empty()

    st.markdown("<div class='section-header'>Alert Log</div>", unsafe_allow_html=True)
    log_ph = st.empty()

_upload_ready = (input_mode == "Upload Video" and uploaded_frames is not None)
_demo_ready   = (input_mode == "Demo Clips")
start = st.button("▶  Start Demo", use_container_width=True, disabled=not (_upload_ready or _demo_ready))

if not start:
    if input_mode == "Upload Video" and not _upload_ready:
        frame_ph.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "padding:60px;text-align:center;color:#8b949e;font-size:1.1rem'>"
            "⬆️ &nbsp; Upload a video file in the sidebar, then press <b>Start Demo</b></div>",
            unsafe_allow_html=True,
        )
    else:
        frame_ph.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            "padding:60px;text-align:center;color:#8b949e;font-size:1.1rem'>"
            "👆 &nbsp; Press <b>Start Demo</b> to begin</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Load clip — unified for both demo clips and uploaded video
# ---------------------------------------------------------------------------
model   = get_model()
gt_mask = None

if input_mode == "Demo Clips":
    clip_dir  = FRAMES_ROOT / clip_id
    mask_path = MASKS_ROOT / f"{clip_id}.npy"
    if not clip_dir.exists():
        st.error(f"Clip not found: {clip_dir}")
        st.stop()
    frame_paths  = _sorted_frame_paths(clip_dir)
    total_frames = len(frame_paths)
    gt_mask      = np.load(mask_path).astype(np.uint8).reshape(-1) if mask_path.exists() else None

    def get_frame(idx: int) -> np.ndarray | None:
        return cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
else:
    total_frames = len(uploaded_frames)  # type: ignore[arg-type]

    def get_frame(idx: int) -> np.ndarray | None:
        return uploaded_frames[idx]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Phase 1: Pre-score all windows (inference happens here, not during playback)
# ---------------------------------------------------------------------------
frame_scores: list[float] = [0.0] * total_frames
score_history: list[dict] = []
alert_log: list[dict]     = []

with st.spinner("Analysing clip… (this takes a few seconds)"):
    progress_bar = st.progress(0, text="Scoring frames…")
    frame_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    last_score = 0.0

    for idx in range(total_frames):
        frame = get_frame(idx)
        if frame is None:
            frame_scores[idx] = last_score
            continue

        frame_buffer.append(frame)

        if len(frame_buffer) == WINDOW_SIZE and idx % WINDOW_STRIDE == 0:
            feat       = frames_to_window_feat(list(frame_buffer))
            last_score = float(model.predict_proba(feat)[0, 1])
            level      = score_to_level(last_score)
            score_history.append({"frame": idx, "score": last_score})

            gt_label = int(gt_mask[idx]) if (gt_mask is not None and idx < len(gt_mask)) else None
            alert_entry: dict = {
                "frame":    idx,
                "time (s)": f"{idx / 25:.1f}",
                "level":    level,
                "score":    f"{last_score:.3f}",
            }
            if gt_mask is not None:
                alert_entry["gt"] = "ANOMALY" if gt_label == 1 else "normal"
            if level in ("MEDIUM", "HIGH"):
                alert_log.append(alert_entry)

        frame_scores[idx] = last_score
        progress_bar.progress((idx + 1) / total_frames,
                              text=f"Scoring frames… {idx + 1}/{total_frames}")

    progress_bar.empty()

# ---------------------------------------------------------------------------
# Phase 2: Smooth playback (no inference — just rendering)
# ---------------------------------------------------------------------------
frame_delay = 1.0 / fps

for frame_idx in range(total_frames):
    frame = get_frame(frame_idx)
    if frame is None:
        continue

    score = frame_scores[frame_idx]
    level = score_to_level(score)
    gt_label = int(gt_mask[frame_idx]) if (gt_mask is not None and frame_idx < len(gt_mask)) else None

    # Video
    vis = draw_overlay(frame, score, level, frame_idx, total_frames,
                       gt_label if show_gt else None)
    frame_ph.image(vis, use_container_width=True)

    # Alert badge
    level_ph.markdown(
        f"<div class='alert-badge alert-{level}'>{LEVEL_ICON[level]} &nbsp; {level}</div>",
        unsafe_allow_html=True,
    )

    # Score card
    score_ph.markdown(
        f"<div class='metric-card'>"
        f"<div class='label'>Anomaly Score</div>"
        f"<div class='value' style='color:{BAR_COLOR[level]}'>{score:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Score bar
    bar_ph.markdown(
        f"<div class='score-bar-wrap'>"
        f"<div class='score-bar-fill' style='width:{score*100:.1f}%;background:{BAR_COLOR[level]}'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Chart — only refresh every N frames to reduce Streamlit rerender cost
    if frame_idx % CHART_REFRESH == 0 and len(score_history) > 1:
        visible = [s for s in score_history if s["frame"] <= frame_idx]
        if visible:
            df = pd.DataFrame(visible).set_index("frame")
            chart_ph.line_chart(df, height=160, use_container_width=True)

    # Alert log — only refresh when new alert added
    if alert_log:
        visible_alerts = [a for a in alert_log if int(a["frame"]) <= frame_idx]
        if visible_alerts:
            log_ph.dataframe(
                pd.DataFrame(visible_alerts[::-1]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        log_ph.markdown(
            "<div style='color:#8b949e;font-size:0.85rem;padding:8px'>No alerts raised — clip is normal.</div>",
            unsafe_allow_html=True,
        )

    time.sleep(frame_delay)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
st.markdown(
    f"<div style='background:#161b22;border:1px solid #238636;border-radius:10px;"
    f"padding:16px;text-align:center;color:#3fb950;font-weight:600;margin-top:12px'>"
    f"✅ &nbsp; Complete — {total_frames} frames · {len(alert_log)} MEDIUM/HIGH alerts</div>",
    unsafe_allow_html=True,
)
