# Setup Guide

Step-by-step instructions to get the system running from scratch on a new machine.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | Any | `git --version` |

A GPU is not required. The system runs on CPU. Inference will be slower (~45–60s per clip) but fully functional.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/DarkMatrix07/crowd-anomaly-detection.git
cd crowd-anomaly-detection
```

---

## Step 2 — Python Environment

Create and activate a virtual environment (recommended):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs: PyTorch (CPU), torchvision, FastAPI, scikit-learn, OpenCV, and all other dependencies.

> **Note:** If you have a GPU and want GPU inference, replace the PyTorch install with the CUDA version from https://pytorch.org/get-started/locally/

---

## Step 3 — Frontend Dependencies

```bash
cd web
npm install
cd ..
```

This installs Next.js 15, React 18, Framer Motion, Recharts, Tailwind CSS, and all other frontend packages.

---

## Step 4 — Verify Model Files

The trained model weights must be present:

```bash
# Check that these files exist
ls artifacts/models/resnet_mlp.pt
ls artifacts/models/shanghaitech_windowed_rf.joblib
```

If they are missing (not committed to the repo due to size), retrain:

```bash
# Train ResNet18+MLP (requires ShanghaiTech dataset — see Step 6)
python scripts/train_resnet_mlp.py

# Train Random Forest baseline
python scripts/train_production_model.py
```

---

## Step 5 — Verify Demo Clips

The 10 demo clips are included in the repository at `data/demo_clips/`. Verify:

```bash
ls data/demo_clips/frames/
# Should show: 01_0130  01_0054  01_0063  02_0128  03_0031
#              04_0001  05_0018  06_0144  07_0005  08_0044

ls data/demo_clips/masks/
# Should show 10 .npy files
```

If the folder is empty, the clips were not pulled (e.g., Git LFS issue). Contact the repo owner.

---

## Step 6 — (Optional) Full Dataset Setup

Only required if you want to retrain models. The demo and dashboard work without the full dataset.

1. Download the ShanghaiTech Campus Dataset from the official source
2. Extract to:
```
data/raw/shanghaitech/shanghaitech/
    training/videos/              # ~330 .avi files
    testing/frames/               # 107 directories of .jpg frames
    testing/test_frame_mask/      # 107 .npy mask files
```

---

## Step 7 — Start the API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Verify it is running:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

---

## Step 8 — Start the Dashboard

Open a new terminal (keep the API server running):

```bash
cd web
npm run dev
```

You should see:
```
▲ Next.js 15.x
- Local: http://localhost:3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Verifying Everything Works

1. Header shows **"Live"** badge (green) — API is reachable
2. Dashboard tab loads metric cards and timeline
3. Click **Live Detection** tab
4. Select any clip from the dropdown
5. Click **Analyze Clip**
6. Wait ~45–60 seconds — video plays back with anomaly scores

---

## Troubleshooting

### API server shows "Module not found" error
```bash
# Make sure you are in the project root, not inside src/
cd crowd-anomaly-detection
uvicorn src.api.app:app --reload --port 8000
```

### Port 3000 already in use
```bash
cd web && npm run dev -- --port 3001
# Then open http://localhost:3001
```

### Port 8000 already in use
```bash
uvicorn src.api.app:app --reload --port 8001
# Then update web/next.config.mjs:
# destination: 'http://127.0.0.1:8001/:path*'
```

### Dashboard shows "Offline" badge
- Check the API server terminal for errors
- Confirm it is running on port 8000
- Try `curl http://127.0.0.1:8000/health` in a terminal

### "ResNet18+MLP model file not found" during analysis
```bash
ls artifacts/models/resnet_mlp.pt
# If missing, retrain: python scripts/train_resnet_mlp.py
```

### Windows: Random Forest hangs during inference
This is a known Windows multiprocessing issue with joblib. It is already handled in the code (`n_jobs=1` at predict time). If it still hangs, restart the API server and try again.

### npm install fails
```bash
# Try clearing the cache
npm cache clean --force
npm install
```

---

## Running Both Servers — Quick Reference

Open two terminals:

**Terminal 1 (API):**
```bash
cd crowd-anomaly-detection
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
uvicorn src.api.app:app --reload --port 8000
```

**Terminal 2 (Dashboard):**
```bash
cd crowd-anomaly-detection/web
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)
