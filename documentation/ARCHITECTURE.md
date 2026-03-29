# System Architecture

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT BROWSER                          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Next.js 15 Dashboard (React)               │   │
│   │                                                         │   │
│   │  ┌──────────────────┐   ┌──────────────────────────┐   │   │
│   │  │  Dashboard Tab   │   │   Live Detection Tab     │   │   │
│   │  │                  │   │                          │   │   │
│   │  │  MetricCard ×3   │   │  Clip Selector           │   │   │
│   │  │  RiskTimeline    │   │  Model Selector          │   │   │
│   │  │  AlertFeed       │   │  VideoOverlay            │   │   │
│   │  │  SidePanel       │   │  AnalysisSummary         │   │   │
│   │  └──────────────────┘   └──────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │  /api/* proxy                        │
└──────────────────────────│──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │  Alert API    │  │  Config API   │  │    Demo Router    │   │
│  │               │  │               │  │                   │   │
│  │  POST /alerts │  │  GET/PUT      │  │  GET /clips       │   │
│  │  GET /alerts  │  │  /config/     │  │  GET /frame/{n}   │   │
│  │  POST /ack    │  │  thresholds   │  │  POST /analyze    │   │
│  │  GET /summary │  │               │  │                   │   │
│  └───────┬───────┘  └───────┬───────┘  └────────┬──────────┘   │
│          │                  │                    │              │
│          ▼                  ▼                    ▼              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │    SQLite     │  │    SQLite     │  │  Inference Engine │   │
│  │  alerts.db    │  │  threshold    │  │                   │   │
│  │               │  │  _config      │  │  ResNet18+MLP     │   │
│  │               │  │               │  │  Random Forest    │   │
│  └───────────────┘  └───────────────┘  └────────┬──────────┘   │
└──────────────────────────────────────────────────│──────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │      Model Artifacts     │
                                    │                          │
                                    │  resnet_mlp.pt           │
                                    │  windowed_rf.joblib      │
                                    └──────────────────────────┘
                                    ┌──────────────────────────┐
                                    │      Demo Clips          │
                                    │                          │
                                    │  data/demo_clips/        │
                                    │  frames/ + masks/        │
                                    └──────────────────────────┘
```

---

## Component Breakdown

### 1. React Frontend (`web/`)

| Component | File | Responsibility |
|-----------|------|---------------|
| Main page | `src/app/page.tsx` | Tab state, polling loop, all dashboard data |
| MetricCard | `src/components/MetricCard.tsx` | Animated count card for LOW/MEDIUM/HIGH |
| RiskTimeline | `src/components/RiskTimeline.tsx` | Recharts AreaChart of score history |
| AlertFeed | `src/components/AlertFeed.tsx` | Paginated alert table with acknowledge buttons |
| SidePanel | `src/components/SidePanel.tsx` | Threshold sliders + acknowledge form |
| LiveDetection | `src/components/LiveDetection.tsx` | Full video analysis + playback UI |
| API client | `src/lib/api.ts` | Typed fetch wrappers for all endpoints |

**Data flow in Dashboard tab:**
```
useEffect (10s interval)
  → fetchSummary() → setAlerts, setTimeline, setLevelCounts
  → fetchThresholds() → setThresholds
  → checkHealth() → setConnected
```

**Data flow in Live Detection tab:**
```
mount → fetchClips() → populate dropdown
Analyze button → analyzeClip(clipId, model)
  → receives {scores, predictions, gt_labels, ...}
  → starts playback loop at chosen FPS
  → each tick: fetch frame image from /demo/clips/{id}/frame/{n}
  → overlay score + risk level + GT label on frame
```

---

### 2. FastAPI Backend (`src/api/`)

| Module | Responsibility |
|--------|---------------|
| `app.py` | App factory, DB init, all dashboard/alert/config endpoints |
| `demo_router.py` | Demo-specific endpoints — clip listing, frame serving, inference |
| `schemas.py` | Pydantic models for request/response validation |

**Request lifecycle:**
```
Browser → Next.js proxy → FastAPI router → handler function
  → SQLite (read/write) or Inference Engine
  → Pydantic serialisation → JSON response → Browser
```

**CORS:** Allows `localhost:3000` and `localhost:3001` (dev). In production, restrict to your domain.

---

### 3. Inference Engine (`src/inference/`)

| Module | Model | Input | Output |
|--------|-------|-------|--------|
| `resnet_mlp_model.py` | ResNet18+MLP | `np.ndarray (N, H, W, 3)` | `float` (score 0–1) |
| `anomaly_model.py` | Random Forest | `np.ndarray (N, H, W, 3)` | `float` (score 0–1) |
| `pipeline.py` | Any | Frame stream | Rolling scores |

Both adapters expose the same interface: `anomaly_model_fn(clip: np.ndarray) -> float`, making them interchangeable.

---

### 4. Database (`artifacts/alerts.db`)

SQLite — single file, no server required.

```
alerts
├── id (PK)
├── timestamp
├── camera_id
├── risk_level
├── score
├── evidence_start / evidence_end
├── acknowledged_by
├── ack_note
└── acknowledged_at

threshold_config
├── id = 1 (single row)
├── profile_name
├── low / medium / high
└── updated_at
```

Schema is initialised on first startup. Migration is handled by `_ensure_alert_columns()` which adds columns if they are missing (backwards compatible with older DB files).

---

### 5. Demo Data (`data/demo_clips/`)

```
data/demo_clips/
├── frames/
│   ├── 01_0130/   337 × JPEG (428×240)
│   ├── 02_0128/   457 × JPEG
│   └── ...        (10 clips total)
└── masks/
    ├── 01_0130.npy   (337 binary labels)
    ├── 02_0128.npy   (457 binary labels)
    └── ...
```

Frames are pre-resized to 428×240 (half the original 856×480) to reduce repository size from ~200MB to 57MB while maintaining sufficient quality for inference.

---

## Deployment Architecture (Current — Single Machine)

```
┌─────────────────────── Developer Machine ───────────────────────┐
│                                                                  │
│   Terminal 1                          Terminal 2                 │
│   uvicorn :8000  ─────────────────►  npm run dev :3000          │
│   (FastAPI)            /api proxy    (Next.js)                   │
│        │                                  │                      │
│        ▼                                  ▼                      │
│   artifacts/alerts.db             Browser at localhost:3000      │
│   artifacts/models/*.pt                                          │
│   data/demo_clips/                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture (Production — Multi Server)

For a real deployment with live cameras:

```
IP Cameras (RTSP)
      │
      ▼
Frame Capture Workers (OpenCV + Redis queue)
      │
      ▼
Inference Workers (GPU, batch processing)
      │
      ▼
Alert API (FastAPI, load balanced)
      │
  ┌───┴───┐
  │       │
  ▼       ▼
PostgreSQL  WebSocket / SSE
(alerts)    (real-time push to dashboard)
              │
              ▼
          React Dashboard (Vercel / Nginx)
```

The current codebase is structured to support this migration:
- The inference adapters are already stateless functions
- The API is already decoupled from the inference pipeline
- The database schema is compatible with PostgreSQL (standard SQL)
- The frontend polling loop can be replaced with WebSocket listeners

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Feature extractor | Frozen ResNet18 | Transfer learning — no need to train from scratch, computationally efficient |
| Temporal modelling | Window aggregation (mean/std/max/delta) | Simple, interpretable, effective — avoids RNN/Transformer complexity for this dataset size |
| API framework | FastAPI | Async, typed, auto-docs, fastest Python API framework |
| Database | SQLite | Zero configuration, single file, sufficient for prototype scale |
| Frontend | Next.js + React | Industry standard, strong ecosystem, handles SSR and client-side state well |
| Proxy | Next.js rewrites | Avoids CORS, keeps backend address configurable, clean separation |
| Demo data | Committed to git (57MB) | Self-contained repo — clone and run, no separate download step |
