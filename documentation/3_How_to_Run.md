# How to Run

## Abnormal Crowd Behaviour Detection System

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 or higher | Download from https://python.org |
| Git Bash | Any recent | For running `.sh` launcher on Windows |
| RAM | 4 GB minimum | 8 GB recommended |
| Storage | 2 GB free | For dataset and model files |
| GPU | Not required | Runs entirely on CPU |

---

## Quick Start (Recommended)

### Option A — Git Bash Launcher (easiest)

1. Open **Git Bash** in the project folder
2. Run:
   ```bash
   bash START_DEMO.sh
   ```
3. A browser window opens automatically
4. Select a clip from the left sidebar and press **▶ Start Demo**

---

## Manual Setup (Step by Step)

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — demo dashboard
- `opencv-python-headless` — video processing
- `scikit-learn` — RandomForest model
- `joblib` — model loading
- `fastapi` + `uvicorn` — API server
- `numpy`, `pandas` — data handling
- `httpx` — HTTP client for dashboard
- `pytest` — test runner

### Step 2 — Verify Installation

```bash
python -c "import streamlit, cv2, joblib, sklearn; print('All OK')"
```

Expected output: `All OK`

### Step 3 — Verify Model Exists

```bash
ls artifacts/models/shanghaitech_windowed_rf.joblib
```

Expected: file exists (~5.3 MB)

---

## Running the Demo

### Interactive Streamlit Demo (Primary Demo)

```bash
python -m streamlit run scripts/crowd_anomaly_demo.py
```

Opens at: http://localhost:8501

**Using the demo:**
1. Select a clip from the sidebar dropdown
   - Clips marked `★ best demo` show the clearest anomaly
   - `05_0018` is a normal clip (score stays green throughout)
2. Adjust playback speed (1–30 fps slider)
3. Toggle "Show ground-truth label" to compare AI predictions vs human annotations
4. Press **▶ Start Demo**
5. Watch the system score each frame in real time

**Upload your own video:**
1. Switch the sidebar toggle to **Upload Video**
2. Upload any `.mp4`, `.avi`, `.mov`, or `.mkv` file
3. Press **▶ Start Demo** once the file is loaded
4. The system runs the same anomaly scoring on your video

---

## Running the Full Production Stack

The full production system consists of three components running simultaneously:

### Terminal 1 — API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

Verify: http://localhost:8000/health → should return `{"status": "ok"}`

### Terminal 2 — Operator Dashboard

```bash
python -m streamlit run dashboard/app.py --server.port 8502
```

Opens at: http://localhost:8502

### Terminal 3 — Live Feed Demo (feeds alerts into the system)

```bash
python scripts/live_feed_demo.py --clip-id 01_0130
```

This replays a real ShanghaiTech clip through the full pipeline and posts alerts to the API, which then appear on the dashboard in real time.

---

## Running Model Training

### Train the Production Model

```bash
python scripts/train_production_model.py \
  --test-ratio 0.20 \
  --window-stride 5 \
  --frame-stride-clips 5 \
  --max-train-videos 0 \
  --n-estimators 300 \
  --classifier rf \
  --model-out artifacts/models/shanghaitech_windowed_rf.joblib \
  --metrics-out artifacts/reports/shanghaitech_windowed_rf_metrics.json
```

**Expected training time:** 5–15 minutes on a standard laptop (CPU only)

**Expected output:**
```
ROC-AUC: 0.8313
PR-AUC:  0.8261
F1:      0.787 (threshold 0.537)
Recall:  0.885
```

### Run Ablation Studies

```bash
python scripts/run_ablations.py
```

Runs all 4 model variants and outputs results to `docs/reports/ablation-results.csv`

---

## Running Tests

```bash
pytest tests/ -v
```

Run a specific test module:
```bash
pytest tests/test_smoke.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Using the API Directly

The FastAPI server provides a full REST API. Access interactive docs at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Example API Calls

**Check health:**
```bash
curl http://localhost:8000/health
```

**Post an alert:**
```bash
curl -X POST http://localhost:8000/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-03-04T12:00:00",
    "camera_id": "cam_01",
    "risk_level": "HIGH",
    "score": 0.91,
    "evidence_window": [0, 30]
  }'
```

**Get recent alerts:**
```bash
curl "http://localhost:8000/alerts?limit=20"
```

**Acknowledge an alert:**
```bash
curl -X POST http://localhost:8000/alerts/1/ack \
  -H "Content-Type: application/json" \
  -d '{"operator_name": "operator-1", "note": "Checked, false alarm"}'
```

**Update thresholds:**
```bash
curl -X PUT http://localhost:8000/config/thresholds \
  -H "Content-Type: application/json" \
  -d '{"profile_name": "strict", "low": 0.35, "medium": 0.65, "high": 0.90}'
```

---

## Configuration

All system parameters are in the `configs/` folder.

### Adjust Risk Thresholds (`configs/risk.yaml`)

```yaml
risk:
  thresholds:
    low: 0.30      # below this = LOW alert
    medium: 0.60   # above this = MEDIUM alert
    high: 0.85     # above this = HIGH alert
  smoothing:
    alpha: 0.35    # higher = more responsive, lower = smoother
```

### Adjust Inference Window (`configs/infer.yaml`)

```yaml
inference:
  clip_length: 30    # number of frames per window
  clip_stride: 10    # frames between window evaluations
```

### Adjust Risk Scoring Weights (`configs/risk.yaml`)

```yaml
risk:
  weights:
    w1: 0.40   # anomaly model score weight
    w2: 0.25   # optical flow instability weight
    w3: 0.20   # crowd density weight
    w4: 0.15   # trend acceleration weight
```

---

## Environment Variables

Copy `.env.example` to `.env` and set values:

```bash
cp .env.example .env
```

```env
DATA_ROOT=./data/raw
DATASET_NAME=shanghaitech
DATASET_SUBSET=part_a
LOG_LEVEL=INFO
OUTPUT_DIR=./artifacts
```

---

## Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: streamlit` | Dependencies not installed | Run `pip install -r requirements.txt` |
| Browser doesn't open | Firewall blocking localhost | Open http://localhost:8501 manually |
| `Model file not found` | Wrong working directory | Run from project root folder |
| Demo runs slowly | Too many background apps | Lower fps slider to 5–8 |
| `UnicodeDecodeError` in BAT | Windows encoding issue | Use `START_DEMO.sh` in Git Bash instead |
| RandomForest hangs on Windows | Multiprocessing bug | Already fixed — model loads with n_jobs=1 |
| Port 8501 already in use | Previous instance running | Kill the old process or use `--server.port 8502` |

---

## File/Folder Requirements

The following must be present for the demo to work:

```
project/
├── artifacts/models/shanghaitech_windowed_rf.joblib   ← required
├── data/raw/shanghaitech/shanghaitech/
│   ├── testing/frames/         ← required for demo clips
│   └── testing/test_frame_mask/  ← required for GT labels
├── scripts/crowd_anomaly_demo.py
└── requirements.txt
```
