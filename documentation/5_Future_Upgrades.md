# Future Upgrades

## Abnormal Crowd Behaviour Detection System

---

## Overview

This document outlines potential future enhancements to the system, categorised by priority, complexity, and expected impact. The current system achieves 83.1% ROC-AUC using a Random Forest model with handcrafted optical flow features. The upgrades below would increase accuracy, real-world applicability, and operational value.

---

## Tier 1 — High Priority, Moderate Effort

### 1. Deep Learning Model Integration

**What:** Replace or augment the Random Forest with a neural network-based anomaly detector.

**Why:** Deep learning models on the ShanghaiTech benchmark typically achieve 90–94% ROC-AUC vs the current 83.1%.

**Options:**

| Model | Expected AUC | Approach |
|-------|-------------|---------|
| Autoencoder (frame reconstruction) | ~85–88% | Trains on normal frames only; anomaly = high reconstruction error |
| ConvLSTM | ~87–90% | Learns spatial + temporal patterns jointly |
| SlowFast / 3D ResNet | ~90–94% | State-of-the-art video understanding backbone |
| Transformer (Video Swin) | ~92–95% | Attention-based temporal modelling |

**Recommended first step:** A convolutional autoencoder trained exclusively on normal clips. It learns what "normal" looks like and flags anything it cannot reconstruct well as anomalous. This is unsupervised, meaning it does not require frame-level anomaly labels for training.

**Implementation effort:** 2–3 weeks. Requires GPU for training but not inference (small model).

---

### 2. RTSP Live Camera Support

**What:** Connect the system to real IP cameras via RTSP streams instead of pre-recorded clips.

**Why:** The production pipeline (`RollingInferencePipeline`) already supports stream input via `src/inference/stream.py`, but the demo currently only uses files. Activating true live streaming would make the system deployable in real venues.

**What needs to be done:**
- Configure RTSP URL in `configs/infer.yaml`
- Add buffering and reconnection logic for dropped streams
- Add multi-camera support (one pipeline instance per camera)
- Test with real IP cameras

**Implementation effort:** 1 week.

---

### 3. Multi-Camera Dashboard

**What:** Extend the Streamlit dashboard to display multiple camera feeds simultaneously in a grid layout.

**Why:** Real deployments have dozens of cameras. The current dashboard shows a single feed. Operators need a bird's-eye view of all cameras at once.

**Features to add:**
- Camera grid view (2×2, 3×3, 4×4 layouts)
- Colour-coded cells (green/yellow/red) based on current alert level
- Click-to-expand individual camera view
- Audio alert when any camera reaches HIGH level

**Implementation effort:** 1–2 weeks.

---

### 4. Optical Flow Visualisation in Demo

**What:** Overlay the actual optical flow vectors on the video frame during playback, showing the direction and magnitude of crowd motion as coloured arrows.

**Why:** Makes the demo much more compelling for presentations. Supervisors and clients can visually see what the AI is "seeing" — the flow arrows speed up and become chaotic as anomaly score rises.

**Implementation:**
- Use `cv2.calcOpticalFlowFarneback` and draw flow vectors with `cv2.arrowedLine`
- Show as HSV colour map or arrow overlay
- Toggle on/off in the demo sidebar

**Implementation effort:** 2–3 days.

---

### 5. Alert Email / SMS Notifications

**What:** Send automated notifications to security staff when a HIGH alert is raised.

**Why:** Operators cannot always watch the dashboard. Push notifications ensure they are alerted even if they are away from their screen.

**Options:**
- Email via SMTP (Python `smtplib`)
- SMS via Twilio API
- WhatsApp via Twilio WhatsApp API
- Microsoft Teams webhook

**Implementation effort:** 2–3 days per notification channel.

---

## Tier 2 — Medium Priority, Higher Effort

### 6. Person Detection and Tracking

**What:** Integrate a person detector (e.g. YOLOv8) to detect individual people in the frame, count them, and track their trajectories.

**Why:** Individual-level analysis enables much richer features:
- Exact head count per zone
- Per-person speed and direction
- Trajectory clustering (are people moving together or scattering?)
- Density maps with spatial resolution

**Expected accuracy gain:** +4–8% AUC from richer features.

**Implementation effort:** 3–4 weeks. YOLOv8 requires GPU for real-time inference.

---

### 7. Zone-Based Analysis

**What:** Divide each camera view into predefined zones (e.g. entrance, exit, platform, concourse). Score each zone independently.

**Why:** A crowd anomaly in one corner of the frame may be masked by normal behaviour in the rest of the frame. Zone-level scoring would be far more sensitive to localised incidents.

**Features:**
- Configurable zone polygons per camera (drawn in a setup UI)
- Independent alert level per zone
- Zone-specific threshold tuning

**Implementation effort:** 2–3 weeks.

---

### 8. Historical Analytics and Reporting

**What:** Add a reporting module that analyses historical alert data to identify patterns.

**Features:**
- Crowd density heatmaps over time
- Peak alert hours / days
- Most problematic cameras / zones
- Weekly / monthly PDF report generation
- Anomaly trend graphs

**Implementation effort:** 2–3 weeks.

---

### 9. Threshold Auto-Calibration

**What:** The system currently uses fixed thresholds (0.50 for MEDIUM, 0.75 for HIGH). Auto-calibration would automatically adjust thresholds per camera based on false positive / false negative feedback.

**How:**
- Operators mark false alarms (acknowledged without action) and missed events
- System adjusts thresholds using a feedback loop (online learning)
- Different cameras get different threshold profiles

**Implementation effort:** 2–3 weeks.

---

### 10. Crowd Flow Simulation Integration

**What:** Integrate a crowd flow simulator (e.g. based on social force model) to predict where a crowd will move in the next 30–60 seconds.

**Why:** Predictive capability — alert before the anomaly reaches critical density, not after.

**Implementation effort:** 4–6 weeks. Research-heavy.

---

## Tier 3 — Advanced / Long-Term

### 11. Edge Deployment (Raspberry Pi / Jetson)

**What:** Package the inference pipeline as a lightweight edge application that runs directly on a camera-mounted compute device.

**Why:** Eliminates network latency, works offline, scales to many cameras without a central server.

**Approach:**
- Convert Random Forest to ONNX format for optimised inference
- Or use `sklearn` directly (already lightweight)
- Package as Docker container for Jetson Nano

**Implementation effort:** 3–4 weeks.

---

### 12. Federated Learning

**What:** Train the model on data from multiple venues without sharing raw video footage (privacy-preserving).

**Why:** Each venue has unique crowd patterns. Federated learning allows the model to improve from real-world data across deployments without violating privacy regulations.

**Implementation effort:** 6–8 weeks. Requires federated learning framework (PySyft, Flower).

---

### 13. Audio Anomaly Detection

**What:** Add a microphone input stream and detect crowd noise anomalies (screaming, sudden loud noise, crowd chanting).

**Why:** Audio and video anomalies are complementary signals. Screaming heard before panic is visible on camera.

**Implementation effort:** 2–3 weeks for basic version.

---

### 14. Web-Based Configuration UI

**What:** Replace the YAML config files with a web-based admin panel where operators can adjust thresholds, zone boundaries, notification settings, and camera configurations without touching any files.

**Implementation effort:** 3–4 weeks.

---

### 15. Mobile App for Alerts

**What:** Native iOS / Android app that receives push notifications and shows live alert status.

**Why:** Security staff are often mobile, not seated at a desktop dashboard.

**Implementation effort:** 6–8 weeks for native apps. 2–3 weeks for a Progressive Web App.

---

## Upgrade Priority Roadmap

```
Phase 1 (1–2 months):
├── RTSP live camera support
├── Optical flow visualisation in demo
├── Alert email/SMS notifications
└── Multi-camera dashboard (basic grid)

Phase 2 (2–4 months):
├── Deep learning model (autoencoder)
├── Zone-based analysis
├── Historical analytics & reporting
└── Threshold auto-calibration

Phase 3 (4–8 months):
├── Person detection & tracking (YOLOv8)
├── Edge deployment (Jetson Nano)
├── Web-based admin UI
└── Mobile alert app

Long-term Research:
├── Federated learning
├── Audio anomaly detection
└── Predictive crowd flow simulation
```

---

## Performance Improvement Forecast

| Upgrade | Expected AUC Improvement | Notes |
|---------|------------------------|-------|
| Autoencoder | +2–5% | Unsupervised, easy to train |
| ConvLSTM | +4–7% | Requires GPU for training |
| SlowFast / Video Swin | +7–11% | SOTA, GPU required |
| Person tracking features | +4–8% | Richer feature set |
| Zone-level analysis | +2–5% | Better localisation |
| Audio fusion | +1–3% | Complementary signal |
| All combined | ~92–96% | Competitive with published SOTA |
