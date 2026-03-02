# Abnormal Crowd Behaviour Detection System
## Executive Summary

**Prepared by:** Manas Chowdary Kannikanti and team, SRM University AP
**Supervisor:** Dr. Surochita Pal
**Delivered to:** Client

---

## What This System Does

This system watches video footage from surveillance cameras and **automatically detects when a crowd starts behaving abnormally** — such as panic, sudden mass movement, bottlenecks, or dangerous congestion — and **raises an early warning before the situation escalates**.

Think of it as an always-on, tireless security analyst watching your camera feeds and alerting operators the moment something unusual begins.

---

## The Problem It Solves

In large public spaces (stations, stadiums, shopping centres, campuses), security staff cannot watch every camera at once. By the time a human operator notices a developing crowd incident, it may already be critical.

This system fills that gap — it watches continuously and raises alerts within seconds of a crowd anomaly beginning.

---

## How It Works (Simple Version)

1. **Camera feed comes in** — the system receives live or recorded video footage.
2. **Every few frames, it analyses motion** — it measures how the crowd is moving: speed, direction, density, and whether the movement is chaotic or orderly.
3. **It scores the scene** — each analysis produces a risk score between 0.0 (completely normal) and 1.0 (highly abnormal).
4. **It raises an alert** — if the score crosses a threshold, the dashboard shows a colour-coded warning.

---

## Alert Levels

| Level | Colour | Score | Meaning | Suggested Action |
|-------|--------|-------|---------|-----------------|
| LOW | Green | 0.0 – 0.50 | Normal crowd behaviour | No action needed |
| MEDIUM | Yellow | 0.50 – 0.75 | Unusual activity detected | Operator should check the feed |
| HIGH | Red | 0.75 – 1.0 | Strong anomaly signal | Immediate attention required |

---

## How Well Does It Work?

The system was trained and tested on the **ShanghaiTech Campus Dataset**, a standard international benchmark for crowd anomaly detection.

| Metric | Result | Plain English |
|--------|--------|---------------|
| Detection Accuracy (ROC-AUC) | **83.1%** | Out of 10 random normal/abnormal scene pairs, it correctly identifies which is abnormal 8.3 times |
| Alert Precision (F1 Score) | **78.7%** | When it raises an alert, it is correct approximately 8 out of 10 times |
| Anomaly Recall | **88.5%** | It catches almost 9 out of every 10 real anomalies that occur |

These results **exceed the project's target threshold** of 80% detection accuracy and 70% F1 score.

---

## What the Client Receives

| Item | Description |
|------|-------------|
| `START_DEMO.bat` | Double-click to launch the live visual demo |
| `dashboard/` | Full operator monitoring dashboard |
| `src/` | System source code |
| `artifacts/models/` | Trained AI model |
| `configs/` | System configuration files |
| `docs/reports/` | Full technical reports and evaluation results |

---

## How to Run the Demo

1. Ensure Python 3.11+ is installed on the machine.
2. Double-click **`START_DEMO.bat`**.
3. A browser window opens automatically.
4. Select a test clip from the dropdown on the left (e.g. *"01_0130 — score peaks at 0.997 ★ best demo"*).
5. Press **▶ Start Demo**.
6. Watch the system score each frame in real time and raise alerts on abnormal clips.

**The demo includes both normal clips** (score stays green throughout) **and abnormal clips** (score rises to MEDIUM/HIGH as the anomaly unfolds), so the difference is immediately visible.

---

## Key Technical Facts (for reference)

- **Model type:** Random Forest classifier with 30-frame temporal window
- **Input features:** 40-dimensional vector per window (motion magnitude, flow variance, directional entropy, crowd density, and temporal acceleration)
- **Training data:** ShanghaiTech Campus Dataset — 330 normal training videos, 60 annotated test clips
- **Inference speed:** Scores a new window every 10 frames (~real time at 25 fps)
- **Platform:** Python 3.11, runs on standard CPU hardware (no GPU required for inference)

---

## Contact

For questions about this system, please contact the project team at SRM University AP.
