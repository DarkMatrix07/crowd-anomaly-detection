# API Reference

Base URL: `http://127.0.0.1:8000`

All endpoints return JSON. The React frontend proxies `/api/*` to this base URL.

---

## Health

### GET /health

Check if the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### GET /status

Returns service status and total alert count.

**Response:**
```json
{
  "service": "up",
  "alerts_count": 31
}
```

---

## Alerts

### POST /alerts

Create a new alert (called by the inference pipeline).

**Request body:**
```json
{
  "timestamp": 1711700000.0,
  "camera_id": "CAM_01",
  "risk_level": "HIGH",
  "score": 0.921,
  "evidence_window": [120, 150]
}
```

| Field | Type | Description |
|-------|------|-------------|
| timestamp | float | Unix timestamp |
| camera_id | string | Camera identifier |
| risk_level | string | "LOW", "MEDIUM", or "HIGH" |
| score | float | Anomaly score 0.0–1.0 |
| evidence_window | [int, int] | Start and end frame indices of the anomalous window |

**Response (201):**
```json
{
  "id": 42,
  "timestamp": 1711700000.0,
  "camera_id": "CAM_01",
  "risk_level": "HIGH",
  "score": 0.921,
  "evidence_window": [120, 150],
  "acknowledged_by": null,
  "ack_note": null,
  "acknowledged_at": null
}
```

---

### GET /alerts?limit=50

List recent alerts, newest first.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Maximum number of alerts to return |

**Response:**
```json
{
  "alerts": [
    {
      "id": 42,
      "timestamp": 1711700000.0,
      "camera_id": "CAM_01",
      "risk_level": "HIGH",
      "score": 0.921,
      "evidence_window": [120, 150],
      "acknowledged_by": "operator-1",
      "ack_note": "Checked — false alarm, students running to class",
      "acknowledged_at": 1711700120.0
    }
  ]
}
```

---

### POST /alerts/{alert_id}/ack

Acknowledge an alert with operator name and note.

**Path parameter:** `alert_id` — integer ID of the alert

**Request body:**
```json
{
  "operator_name": "operator-1",
  "note": "Confirmed anomaly — security dispatched"
}
```

**Response (200):** Full alert record with updated acknowledgement fields.

**Response (404):** Alert not found.

---

## Dashboard

### GET /dashboard/summary?limit=100&camera_id=CAM_01

Aggregated dashboard data: alert list, timeline, and level counts.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 100 | Max alerts to return |
| camera_id | string | null | Filter by specific camera (omit for all cameras) |

**Response:**
```json
{
  "alerts": [...],
  "timeline": [
    {
      "timestamp": 1711700000.0,
      "score": 0.921,
      "risk_level": "HIGH",
      "camera_id": "CAM_01"
    }
  ],
  "level_counts": {
    "low": 18,
    "medium": 9,
    "high": 4
  }
}
```

---

## Configuration

### GET /config/thresholds

Get current alert threshold configuration.

**Response:**
```json
{
  "profile_name": "default",
  "low": 0.30,
  "medium": 0.60,
  "high": 0.85
}
```

---

### PUT /config/thresholds

Update alert thresholds. Values are automatically sorted ascending before saving.

**Request body:**
```json
{
  "profile_name": "strict",
  "low": 0.20,
  "medium": 0.45,
  "high": 0.70
}
```

**Response (200):** Updated threshold profile (values sorted ascending).

---

## Demo

### GET /demo/clips

List all available demo clips.

**Response:**
```json
{
  "clips": [
    {
      "id": "01_0130",
      "description": "Scene 01 — Sudden crowd rush (peaks 0.997)",
      "frame_count": 337,
      "has_ground_truth": true
    },
    {
      "id": "02_0128",
      "description": "Scene 02 — Dense crowd anomaly (55% anomaly)",
      "frame_count": 457,
      "has_ground_truth": true
    }
  ]
}
```

**Response (503):** Demo frames not found on disk.

---

### GET /demo/clips/{clip_id}/frame/{frame_idx}

Serve a single JPEG frame from a demo clip.

**Path parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| clip_id | string | Clip identifier e.g. "01_0130" |
| frame_idx | int | Zero-based frame index |

**Response:** JPEG image (`Content-Type: image/jpeg`)

**Response (404):** Clip or frame not found.

---

### POST /demo/clips/{clip_id}/analyze

Run full inference on a demo clip and return per-frame scores.

**Path parameter:** `clip_id` — e.g. "01_0130"

**Request body:**
```json
{
  "model": "resnet",
  "window_size": 30,
  "stride": 10
}
```

| Field | Type | Options | Default | Description |
|-------|------|---------|---------|-------------|
| model | string | "resnet", "rf" | "resnet" | Which model to use |
| window_size | int | — | 30 | Frames per window |
| stride | int | — | 10 | Step between windows |

**Response:**
```json
{
  "clip_id": "01_0130",
  "model": "resnet",
  "frame_count": 337,
  "scores": [0.12, 0.13, 0.15, ..., 0.997],
  "predictions": [0, 0, 0, ..., 1],
  "gt_labels": [0, 0, 0, ..., 1],
  "peak_score": 0.997,
  "mean_score": 0.412,
  "anomaly_frames": 94,
  "accuracy": 0.913,
  "gt_anomaly_pct": 27.9,
  "roc_auc": 0.9466
}
```

| Field | Description |
|-------|-------------|
| scores | Per-frame anomaly score array, length = frame_count |
| predictions | Per-frame binary prediction (1 = anomaly) at threshold 0.5 |
| gt_labels | Ground truth labels (null if no mask file exists) |
| peak_score | Maximum score across all frames |
| mean_score | Average score across all frames |
| anomaly_frames | Count of frames with score >= 0.5 |
| accuracy | Fraction of frames correctly classified vs ground truth |
| gt_anomaly_pct | Percentage of ground truth frames that are anomalous |
| roc_auc | Area under ROC curve vs ground truth (if available) |

**Response (404):** Clip not found.
**Response (500):** Model file missing or inference error — check server logs.

---

## Error Format

All errors follow this format:

```json
{
  "detail": "Alert not found"
}
```
