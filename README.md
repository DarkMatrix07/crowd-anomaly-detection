# Abnormal Crowd Behaviour Detection

Early-warning system for detecting abnormal crowd behaviour in surveillance footage using deep learning and traditional machine learning. Built on the ShanghaiTech Campus dataset.

**Capstone Project — SRM University AP · 2026**

---

## Results

| Model | Approach | ROC-AUC | Accuracy |
|-------|----------|---------|----------|
| **ResNet18 + MLP** | Transfer learning (DL) | **0.9715** | **91.96%** |
| Random Forest (W=30) | Hand-crafted features | 0.8313 | 76.1% |

Production model: **ResNet18 + MLP** (satisfies DL requirement, highest performance)

---

## Architecture

```
Video frames
    │
    ▼
ResNet18 (frozen, ImageNet pretrained)
    │  512-d feature vector per frame
    ▼
Temporal Window (30 frames)
    │  mean + std + max + delta  →  2048-d vector
    ▼
MLP Classifier (2048 → 256 → 64 → 1)
    │
    ▼
Anomaly Score [0, 1]  →  LOW / MEDIUM / HIGH alert
```

---

## Project Structure

```
.
├── web/                        # React/Next.js dashboard (primary UI)
│   └── src/
│       ├── app/page.tsx        # Main dashboard + tab navigation
│       ├── components/
│       │   ├── LiveDetection.tsx
│       │   ├── AlertFeed.tsx
│       │   ├── RiskTimeline.tsx
│       │   ├── MetricCard.tsx
│       │   └── SidePanel.tsx
│       └── lib/api.ts          # Typed API client
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI app (alerts, thresholds, dashboard)
│   │   ├── demo_router.py      # Demo endpoints (clips, frames, inference)
│   │   └── schemas.py
│   ├── inference/
│   │   ├── resnet_mlp_model.py # ResNet18+MLP inference adapter
│   │   ├── anomaly_model.py    # Random Forest inference adapter
│   │   └── pipeline.py        # Rolling inference pipeline
│   ├── models/
│   │   └── resnet_mlp.py       # WindowMLP architecture
│   └── features/
│       └── anomaly_features.py # Feature extraction (optical flow + HOG)
├── scripts/
│   ├── train_resnet_mlp.py     # Train ResNet18+MLP
│   ├── evaluate_resnet_mlp.py  # Evaluate ResNet18+MLP
│   └── crowd_anomaly_demo.py   # Streamlit demo app
├── artifacts/models/           # Trained model weights
├── data/demo_clips/            # 10 pre-selected demo clips (428x240, 57 MB)
│   ├── frames/                 # JPEG frames per clip
│   └── masks/                  # Ground truth anomaly masks (.npy)
├── docs/reports/               # Evaluation reports and ablation study
├── configs/                    # Training/inference config files
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.11+, Node.js 18+

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd web && npm install
```

---

## Running the System

### 1. Start the API server

```bash
uvicorn src.api.app:app --reload --port 8000
```

### 2. Start the dashboard

```bash
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Dashboard

The React dashboard has two tabs:

**Dashboard tab**
- Live alert feed with severity badges (LOW / MEDIUM / HIGH)
- Risk score timeline chart
- Configurable alert thresholds with profile presets (default / strict / relaxed)
- Operator alert acknowledgement with notes

**Live Detection tab**
- Select from 10 demo clips across 6 different campus scenes
- Choose model: ResNet18+MLP (0.97 AUC) or Random Forest (0.83 AUC)
- Frame-by-frame playback with live anomaly score overlay
- Ground truth comparison and per-clip ROC-AUC

---

## Training

```bash
# Train primary DL model (ResNet18 + MLP)
python scripts/train_resnet_mlp.py

# Train Random Forest baseline
python scripts/train_production_model.py
```

Requires the ShanghaiTech dataset at:
```
data/raw/shanghaitech/shanghaitech/
    training/videos/              # ~330 normal .avi clips
    testing/frames/               # 107 test clip directories
    testing/test_frame_mask/      # Ground truth .npy masks
```

---

## Evaluation

```bash
# Evaluate ResNet18+MLP
python scripts/evaluate_resnet_mlp.py
```

Full results, ablation study, and baseline comparisons are in `docs/reports/`.

---

## Demo Clips

10 clips from 6 campus scenes are included in `data/demo_clips/` (resized to 428x240):

| Clip | Scene | Description |
|------|-------|-------------|
| 01_0130 | Scene 01 | Sudden crowd rush — peaks at 0.997 |
| 02_0128 | Scene 02 | Dense crowd anomaly — 55% anomaly frames |
| 01_0063 | Scene 01 | Fast-escalating aggression |
| 01_0054 | Scene 01 | Early warning, slow build-up |
| 05_0018 | Scene 05 | Normal pedestrian flow (baseline) |
| 03_0031 | Scene 03 | Loitering and irregular movement |
| 04_0001 | Scene 04 | Chaotic crowd scatter — 52% anomaly |
| 06_0144 | Scene 06 | High-density anomaly — 58% anomaly |
| 07_0005 | Scene 07 | Sporadic abnormal behaviour |
| 08_0044 | Scene 08 | Running and chasing incident |

---

## Dataset

[ShanghaiTech Campus Dataset](https://svip-lab.github.io/dataset/campus_dataset.html) — 13 scenes, 437 normal training videos, 107 annotated test clips with frame-level ground truth masks.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 18, Tailwind CSS, Framer Motion, Recharts |
| Backend | FastAPI, SQLite |
| ML | PyTorch, torchvision (ResNet18), scikit-learn |
| Features | OpenCV (optical flow), HOG descriptors |

---

## Team

Capstone project — SRM University AP, 2026
Supervisor: Dr. Surochita Pal
