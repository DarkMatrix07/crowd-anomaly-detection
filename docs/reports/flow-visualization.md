# Optical Flow Visualisations

Generated figures are saved under `docs/reports/figures/flow/`.

Each figure is a three-panel strip:
- **Left panel** — original camera frame with scene label
- **Centre panel** — HSV optical flow map (hue = direction, brightness = speed)
- **Right panel** — flow arrow overlay (arrows show motion direction and magnitude)

---

## Figures

### 1. Normal Scene
**File:** `normal_05_0018_frame060.png`
**Source:** Clip `05_0018`, frame 60
**Alert level:** LOW

Orderly pedestrian movement. Flow magnitude is very low and uniform.
The HSV map shows near-black (minimal motion). Arrows are sparse and short.

| Metric | Value |
|--------|-------|
| Mean flow magnitude | 0.039 |
| Max flow magnitude | 1.493 |

---

### 2. Building / Pre-Anomaly Phase
**File:** `congestion_01_0130_frame040.png`
**Source:** Clip `01_0130`, frame 40 (anomaly onset at frame 127)
**Alert level:** LOW → MEDIUM (transitioning)

Crowd density is rising. Flow magnitude has increased noticeably but motion is
still somewhat directional. The HSV map shows patchy colour — some areas moving,
others static. Arrows are longer than normal but still somewhat ordered.

| Metric | Value |
|--------|-------|
| Mean flow magnitude | 0.116 |
| Max flow magnitude | 7.643 |

---

### 3. Anomaly Peak — Panic / Chaotic Motion
**File:** `anomaly_01_0130_frame160.png`
**Source:** Clip `01_0130`, frame 160 (deep into anomaly zone)
**Alert level:** HIGH (model score 0.997)

Full panic / chaotic dispersal. Flow magnitude is 14× higher than the normal scene.
The HSV map is bright and multi-coloured — pixels moving in all directions simultaneously
(high directional entropy). Arrows are long, dense, and point in conflicting directions.

| Metric | Value |
|--------|-------|
| Mean flow magnitude | 0.550 |
| Max flow magnitude | 10.580 |

---

### 4. Side-by-Side Comparison (all three scenes)
**File:** `comparison_all_scenes.png`

All three scenes stacked vertically with a colour-coded sidebar:
- 🟢 Green bar — Normal
- 🟡 Orange bar — Building / pre-anomaly
- 🔴 Red bar — Anomaly peak

This is the recommended slide for supervisor and client presentations as it shows
the full progression from normal → escalating → panic in one image.

---

## Key Takeaway for Presentation

The optical flow visualisation makes the AI's reasoning transparent:

> "When mean flow magnitude rises from 0.04 to 0.55 and the HSV map
> lights up with conflicting colours, the model has observed the same
> motion chaos a human security officer would see — and raises a HIGH alert."
