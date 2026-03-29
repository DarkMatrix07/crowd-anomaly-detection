# Project Overview

## Abnormal Crowd Behaviour Detection System

---

## What is this project?

This is an AI-powered surveillance system that watches video footage from campus cameras and automatically detects when something abnormal is happening in a crowd — such as a fight, a stampede, people running, or any unusual gathering behaviour. When it detects something, it raises an alert so a human operator can review and respond.

---

## Why does this matter?

Traditional CCTV surveillance requires a human to watch screens at all times. This is:
- Expensive (requires dedicated staff 24/7)
- Unreliable (humans miss events due to fatigue or distraction)
- Slow (response time depends on when someone notices)

An automated system can monitor every frame of every camera simultaneously, flag suspicious activity instantly, and let operators focus only on confirmed threats.

---

## Dataset

**ShanghaiTech Campus Dataset**
- Recorded across 13 different scenes on a real university campus
- 437 normal training videos (~330 .avi clips)
- 107 test clips with hand-annotated frame-level ground truth labels
- Anomaly types: running, fighting, chasing, loitering, jumping, throwing objects, cycling in pedestrian zones, sudden crowd gathering

The ground truth labels (0 = normal, 1 = anomaly) were manually annotated by the dataset authors — every single frame of every test clip was labeled by hand.

---

## Models Built

### 1. ResNet18 + MLP (Primary — Production Model)
- **Type:** Supervised deep learning
- **ROC-AUC:** 0.9715
- **Accuracy:** 91.96%
- **How:** Pretrained ResNet18 extracts spatial features from each frame. A 30-frame temporal window is aggregated and classified by a small MLP neural network.

### 2. Random Forest W=30 (Baseline)
- **Type:** Supervised machine learning
- **ROC-AUC:** 0.8313
- **How:** Optical flow + HOG features are extracted manually from frames and fed into a Random Forest classifier over a 30-frame window.

Both models are available in the live demo for side-by-side comparison.

---

## System Components

| Component | Description |
|-----------|-------------|
| **React Dashboard** | Web-based monitoring UI with live alerts, risk timeline, threshold controls |
| **Live Detection** | Interactive demo — select a video clip, run inference, watch frame-by-frame |
| **FastAPI Backend** | REST API serving alerts, thresholds, dashboard data, and demo inference |
| **SQLite Database** | Stores alerts, acknowledgements, and threshold configuration |
| **Inference Pipeline** | Rolling window inference that scores video frames in real time |

---

## Key Results

| Metric | ResNet18 + MLP | Random Forest |
|--------|---------------|---------------|
| ROC-AUC | **0.9715** | 0.8313 |
| PR-AUC | **0.9515** | 0.8261 |
| Accuracy | **91.96%** | — |
| Anomaly Recall | **99%** | 88.5% |
| Anomaly Precision | 85% | — |
| F1 Score | **0.92** | 0.787 |

---

## Ablation Study (Random Forest Variants)

To understand which design choices matter, we tested 4 variants:

| Variant | ROC-AUC | Observation |
|---------|---------|-------------|
| RF W=30 (production) | 0.8313 | Best RF configuration |
| GBT W=30 | 0.8180 | Gradient boosting slightly weaker |
| RF W=30, no optical flow | 0.8020 | Flow features add ~3 AUC points |
| RF W=15 | 0.7450 | Shorter window loses 8.6 points — temporal context is critical |

**Conclusion:** 30-frame windows with optical flow features are essential for strong performance.

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | PyTorch, torchvision (ResNet18) |
| Machine Learning | scikit-learn (RandomForest) |
| Feature Extraction | OpenCV (optical flow), HOG descriptors |
| API | FastAPI, SQLite |
| Frontend | Next.js 15, React 18, Tailwind CSS, Framer Motion, Recharts |

---

## Project Timeline

| Phase | Deliverable |
|-------|-------------|
| Phase 1 | Infrastructure setup — FastAPI, database, inference pipeline |
| Phase 2 | Random Forest baseline — feature engineering, training, evaluation |
| Phase 3 | Ablation study — 4 RF variants compared |
| Phase 4 | ResNet18+MLP — DL model training and evaluation |
| Phase 5 | React dashboard — professional monitoring UI |
| Phase 6 | Demo system — 10 annotated clips, live inference on dashboard |

---

*Capstone Project — SRM University AP, 2026 | Supervisor: Dr. Surochita Pal*
