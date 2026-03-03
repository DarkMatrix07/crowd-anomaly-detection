# About the Project

## Abnormal Crowd Behaviour Detection System

---

## Overview

This project is a **real-time crowd anomaly detection system** built as a Capstone project at SRM University AP. It watches video footage from surveillance cameras and automatically detects when a crowd starts behaving abnormally — such as panic, sudden mass movement, dangerous congestion, or bottlenecks — and raises an early warning alert before the situation escalates.

The system is designed to operate continuously on standard CPU hardware, making it practical for deployment in public venues without requiring expensive GPU infrastructure.

---

## Background and Motivation

In large public spaces — train stations, stadiums, shopping centres, university campuses, airports — security staff cannot watch every camera feed at the same time. Human attention is limited and fatigue sets in over long shifts. By the time an operator notices a developing crowd incident on a monitor, the situation may already be critical or beyond control.

Traditional alarm systems rely on manually triggered buttons or simple motion detectors, which cannot understand the difference between normal busy crowd movement and genuinely dangerous crowd dynamics.

This system fills that gap. It acts as a tireless, always-on AI analyst that watches every camera feed simultaneously, scores each scene in real time, and raises a colour-coded alert the moment abnormal behaviour begins — giving operators precious extra seconds to respond.

---

## Academic Context

| Field | Detail |
|-------|--------|
| Institution | SRM University AP |
| Student Lead | Manas Chowdary Kannikanti |
| Team Size | 4 members |
| Supervisor | Dr. Surochita Pal |
| Type | Capstone / Final Year Project |
| Dataset | ShanghaiTech Campus Dataset (international benchmark) |

---

## Project Goals

1. **Detect** abnormal crowd behaviour automatically from video footage
2. **Score** each scene continuously on a scale of 0.0 (normal) to 1.0 (highly abnormal)
3. **Alert** security operators with colour-coded warnings (LOW / MEDIUM / HIGH)
4. **Demonstrate** the system running live on real surveillance test clips
5. **Evaluate** performance against international benchmarks

---

## Performance Targets vs. Achieved Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Detection Accuracy (ROC-AUC) | ≥ 80% | **83.1%** | PASSED |
| Alert Precision (F1 Score) | ≥ 70% | **78.7%** | PASSED |
| Anomaly Recall | — | **88.5%** | Excellent |
| Platform | CPU only | CPU only | Met |

All project performance gates have been cleared.

---

## What the System Can Detect

The system is trained to identify the following types of crowd anomalies, all present in the ShanghaiTech Campus Dataset:

- **Panic and sudden mass movement** — large groups of people suddenly moving in the same direction at high speed
- **Fighting or violent altercations** — localised intense motion against a background of normal movement
- **Sudden crowd dispersion** — people rapidly spreading outward from a central point
- **Dangerous congestion / bottlenecks** — crowd density spiking in confined areas
- **Irregular or chaotic motion** — loss of orderly directional flow (high directional entropy)
- **Running** — abnormally fast motion across the scene

---

## Alert Levels

| Level | Colour | Score Range | Meaning | Recommended Action |
|-------|--------|-------------|---------|-------------------|
| LOW | Green | 0.00 – 0.50 | Normal crowd behaviour | No action needed |
| MEDIUM | Yellow | 0.50 – 0.75 | Unusual activity detected | Operator should check the feed |
| HIGH | Red | 0.75 – 1.00 | Strong anomaly signal | Immediate attention required |

---

## System Architecture (High Level)

```
Video Input (file or RTSP stream)
        │
        ▼
  Frame Extractor
  (cv2.VideoCapture)
        │
        ▼
  Rolling Inference Pipeline
  ┌─────────────────────────┐
  │ Buffer 30 frames        │
  │ Extract motion features │
  │ Score with RF model     │
  │ Compute risk score      │
  │ Apply smoothing         │
  │ Apply hysteresis        │
  └─────────────────────────┘
        │
        ▼
  FastAPI Alert Server          Streamlit Dashboard
  (SQLite persistence)    ───►  (Operator monitoring)
        │
        ▼
  Alert: LOW / MEDIUM / HIGH
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Computer Vision | OpenCV (cv2) |
| Machine Learning | scikit-learn (RandomForest) |
| Feature Extraction | Farneback Optical Flow |
| API Server | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite |
| Model Storage | joblib |
| Configuration | YAML |

---

## Dataset

The system was trained and tested on the **ShanghaiTech Campus Dataset**, which is one of the most widely used international benchmarks for crowd anomaly detection research.

| Dataset Property | Value |
|-----------------|-------|
| Normal training videos | 330 |
| Test clips (annotated) | 60 |
| Scenes | 13 different campus locations |
| Ground truth | Frame-level binary masks (.npy files) |
| Anomaly types | Running, fighting, panic, dispersion, theft |

---

## Key Design Decisions

### Why Random Forest instead of Deep Learning?

A Random Forest model with handcrafted optical flow features was chosen over deep learning (CNN, LSTM, etc.) for the following reasons:

1. **No GPU required** — runs on standard CPU hardware
2. **Real-time inference** — scores a 30-frame window in ~100ms
3. **Interpretable** — feature importances are available and explainable
4. **Strong performance** — 0.831 ROC-AUC exceeds the project gate
5. **Stable** — no training instability, no overfitting risk from small datasets
6. **Deployable** — 5.3 MB model file, no CUDA dependencies

### Why Temporal Windows?

Single-frame analysis is too noisy and produces excessive false alarms (walking looks similar to running at the pixel level). By analysing 30-frame windows (~1.2 seconds of video), the system captures:
- How crowd speed is changing over time
- Whether motion is accelerating (panic signature)
- Whether directional flow is becoming chaotic

---

## Deliverables

| Item | Description |
|------|-------------|
| `START_DEMO.sh` | One-click demo launcher (Git Bash) |
| `START_DEMO.bat` | Windows CMD demo launcher |
| `scripts/crowd_anomaly_demo.py` | Interactive Streamlit demo |
| `src/` | Full production source code |
| `artifacts/models/` | Trained model files |
| `configs/` | System configuration |
| `docs/reports/` | Technical evaluation reports |
| `documentation/` | This documentation folder |
