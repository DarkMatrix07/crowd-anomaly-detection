# How It Works

A technical walkthrough of the full system from raw video to alert.

---

## 1. Data Flow Overview

```
Raw Video (frames)
       │
       ▼
┌──────────────────┐
│  Frame Extraction │  OpenCV reads frames, resizes to 320x240
└──────────────────┘
       │
       ▼
┌──────────────────────┐
│  Feature Extraction   │  ResNet18 extracts 512-d spatial features per frame
└──────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Temporal Window (W=30)   │  30 consecutive frames → mean/std/max/delta → 2048-d
└──────────────────────────┘
       │
       ▼
┌──────────────────┐
│  MLP Classifier   │  2048 → 256 → 64 → 1  (sigmoid output)
└──────────────────┘
       │
       ▼
  Anomaly Score [0, 1]
       │
       ▼
┌──────────────────────┐
│  Risk Thresholding    │  Score → LOW / MEDIUM / HIGH
└──────────────────────┘
       │
       ▼
┌──────────────────┐
│  Alert Created    │  Stored in SQLite via FastAPI
└──────────────────┘
       │
       ▼
  Dashboard (React)
```

---

## 2. Feature Extraction (ResNet18)

ResNet18 is a convolutional neural network pretrained on ImageNet. We use it as a **frozen feature extractor** — its weights are not updated during training.

For each frame:
1. Resize to 224×224 (ResNet input size)
2. Normalise using ImageNet mean and std
3. Pass through ResNet18 up to the global average pooling layer
4. Extract the 512-dimensional output vector

This gives us a rich 512-d representation of each frame's visual content — edges, shapes, textures, spatial structure — without needing to train a CNN from scratch.

```python
# Pseudocode
resnet = torchvision.models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
resnet.eval()

for frame in clip:
    tensor = preprocess(frame)          # 3x224x224
    features = resnet(tensor)           # 512-d
```

---

## 3. Temporal Window Aggregation

A single frame's features do not capture motion. We look at **30 consecutive frames** (approximately 1 second of video) and summarise them:

For each of the 512 feature dimensions, we compute:
- **Mean** — average activation across 30 frames
- **Std** — variability across 30 frames
- **Max** — peak activation
- **Delta** — difference between last and first frame (direction of change)

This gives `512 × 4 = 2048` dimensions per window, encoding both the visual content and its temporal dynamics.

```python
window = features[start : start + 30]   # shape: (30, 512)
agg = np.concatenate([
    window.mean(axis=0),   # 512
    window.std(axis=0),    # 512
    window.max(axis=0),    # 512
    window[-1] - window[0] # 512
])  # → 2048-d
```

---

## 4. MLP Classifier (WindowMLP)

The 2048-d window vector is passed through a small MLP:

```
Input: 2048-d
  → Linear(2048, 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256, 64)   → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64, 1)     → Sigmoid
Output: score ∈ [0, 1]
```

Training details:
- Loss: Binary Cross Entropy with `pos_weight` to handle class imbalance (more normal frames than anomaly frames)
- Optimiser: Adam, lr=1e-3
- Labels: 1 if the window's centre frame is annotated as anomaly, 0 otherwise
- Normal data: from `training/videos/` (all normal)
- Anomaly data: from `testing/frames/` with ground truth masks

---

## 5. Rolling Inference Pipeline

During inference, windows slide across the clip with a stride:

```
Frames:    [0, 1, 2, ... 456]
Window 1:  [0 ... 29]  → score → assigned to centre frame 15
Window 2:  [10 ... 39] → score → assigned to centre frame 25
Window 3:  [20 ... 49] → score → assigned to centre frame 35
...
```

Each frame may be covered by multiple windows. Scores are averaged per frame. Frames not covered by any window inherit the nearest scored frame's value.

This produces a per-frame score array of length N (number of frames in the clip).

---

## 6. Risk Thresholding

Per-frame scores are mapped to risk levels using configurable thresholds:

| Score Range | Risk Level |
|-------------|-----------|
| 0.00 – 0.30 | LOW |
| 0.30 – 0.60 | MEDIUM |
| 0.60 – 0.85 | HIGH |
| 0.85 – 1.00 | HIGH (critical) |

Thresholds are stored in the SQLite database and can be updated at runtime via the dashboard without restarting the server.

Three presets are available:
- **Default:** 0.30 / 0.60 / 0.85
- **Strict:** lower thresholds — flags more events, higher false positive rate
- **Relaxed:** higher thresholds — fewer alerts, may miss borderline events

---

## 7. API Layer (FastAPI)

The FastAPI backend exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/alerts` | GET/POST | List or create alerts |
| `/alerts/{id}/ack` | POST | Acknowledge an alert |
| `/dashboard/summary` | GET | Aggregated counts + timeline |
| `/config/thresholds` | GET/PUT | Read or update thresholds |
| `/demo/clips` | GET | List available demo clips |
| `/demo/clips/{id}/frame/{n}` | GET | Serve a single JPEG frame |
| `/demo/clips/{id}/analyze` | POST | Run full inference on a clip |

The Next.js frontend proxies all `/api/*` requests to FastAPI at `127.0.0.1:8000`, so the browser never calls the backend directly.

---

## 8. Database Schema

Two tables in SQLite (`artifacts/alerts.db`):

**alerts**
```sql
id              INTEGER PRIMARY KEY
timestamp       REAL
camera_id       TEXT
risk_level      TEXT  -- LOW | MEDIUM | HIGH
score           REAL
evidence_start  INTEGER  -- first frame of the anomalous window
evidence_end    INTEGER  -- last frame of the anomalous window
acknowledged_by TEXT
ack_note        TEXT
acknowledged_at REAL
```

**threshold_config**
```sql
id           INTEGER PRIMARY KEY  -- always 1 (single row)
profile_name TEXT
low          REAL
medium       REAL
high         REAL
updated_at   REAL
```

---

## 9. React Dashboard

The frontend is a Next.js 15 application with two main views:

**Dashboard tab** (`page.tsx`)
- Polls the API every 10 seconds
- Renders metric cards (LOW/MEDIUM/HIGH counts) with Framer Motion animations
- Recharts AreaChart for the risk score timeline
- AlertFeed table with per-row acknowledge button
- SidePanel with threshold sliders and acknowledge form

**Live Detection tab** (`LiveDetection.tsx`)
- Fetches clip list from `/api/demo/clips` on mount
- On "Analyze", POSTs to `/api/demo/clips/{id}/analyze` with model choice
- Plays back frames by fetching `/api/demo/clips/{id}/frame/{n}` at the chosen FPS
- Overlays the score, risk level, GT label, and progress bar on each frame
- Shows live stats sidebar and score history strip during playback
- Shows full analysis summary with frame-by-frame chart and GT comparison after playback

---

## 10. Random Forest Baseline (How it Differs)

The Random Forest uses manually engineered features instead of deep features:

1. **Optical Flow** — Lucas-Kanade method computes motion vectors between consecutive frames. From each frame pair, extract: mean flow magnitude, max flow magnitude, flow direction histogram (8 bins), flow variance.

2. **HOG (Histogram of Oriented Gradients)** — Captures edge and shape information from each frame.

3. **Window Aggregation** — Same mean/std/max/delta aggregation over 30 frames, giving a 40-dimensional feature vector.

4. **Random Forest Classifier** — 100 trees, trained on the aggregated feature vectors with binary labels.

The RF is faster at inference but less accurate (0.83 vs 0.97 AUC) because hand-crafted features capture less information than deep features from ResNet18.

---

## 11. Evaluation Methodology

Test set: 107 clips from ShanghaiTech with frame-level ground truth masks.

For each clip:
1. Run rolling inference → per-frame score array
2. Compare against ground truth mask
3. Compute: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

Evaluation is clip-level 80/20 split (not frame-level) to avoid data leakage between similar scenes.

The ROC curve is computed by sweeping the decision threshold from 0 to 1 and computing TPR vs FPR at each point. AUC summarises the area under this curve — higher is better, independent of the chosen threshold.
