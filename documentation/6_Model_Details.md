# Model Details — How It Works and How It Detects Anomalies

## Abnormal Crowd Behaviour Detection System

---

## Overview

The production model is a **Random Forest classifier** that operates on **40-dimensional temporal window features** derived from 30 consecutive video frames. It outputs an anomaly probability between 0.0 and 1.0 for each window.

This document explains the full detection pipeline from raw video pixels to a final risk alert — every stage, every decision, and every number.

---

## The Big Picture

```
Raw Video Frames
      │
      ▼
[1] Frame-Level Feature Extraction (10 features per frame)
      │  ├─ Intensity features (pixel statistics)
      │  └─ Optical Flow features (motion analysis)
      ▼
[2] Temporal Window Aggregation (30 frames → 40-d vector)
      │  ├─ Mean across frames
      │  ├─ Standard deviation across frames
      │  ├─ Maximum across frames
      │  └─ Delta (last third minus first third)
      ▼
[3] Random Forest Classification (40-d vector → probability)
      │  ├─ 300 decision trees vote
      │  └─ Output: anomaly probability [0.0, 1.0]
      ▼
[4] Risk Score Fusion (anomaly + flow + density + trend)
      ▼
[5] Exponential Smoothing (reduce noise)
      ▼
[6] Hysteresis (prevent alert flickering)
      ▼
Final Alert: LOW / MEDIUM / HIGH
```

---

## Stage 1: Frame-Level Feature Extraction

### What Happens

For every video frame, 10 scalar numbers are extracted. These capture both the appearance of the frame and the motion between consecutive frames.

### Feature 1: Mean Pixel Intensity

```python
mean_intensity = np.mean(frame_gray)
```

The average brightness of the entire frame (grayscale). In crowd scenes, a sudden drop (scene darkening, crowd blocking light) or spike can indicate unusual activity. Typical range: 50–200 (out of 255).

---

### Feature 2: Standard Deviation of Pixel Intensity

```python
std_intensity = np.std(frame_gray)
```

How spread out the pixel values are. A high standard deviation means high contrast — sharp edges, clear clothing boundaries, visible crowd structure. A low value means a blurry or featureless scene. Crowds in panic tend to create high-contrast, highly varied scenes.

---

### Feature 3: Laplacian Variance (Sharpness / Edge Energy)

```python
lap_var = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
```

The Laplacian operator computes the second derivative of intensity — it responds strongly to edges and fine textures. Its variance measures how many sharp edges are present in the frame. In crowd scenes, a dense, fast-moving crowd produces many edges (bodies, limbs, clothing). This feature rises sharply during fights or running events.

**Why it works:** A running or fighting crowd produces many rapidly changing edges, which the Laplacian captures as high variance. A static or slowly moving crowd has fewer and more stable edges.

---

### Feature 4: Occupancy (Crowd Density Proxy)

```python
occupancy = np.mean(frame_gray > threshold)  # threshold ~ 0.2 * 255
```

The fraction of pixels that are "bright enough" to likely belong to a person rather than background. This is a simple crowd density estimate: more people visible = higher occupancy. When a crowd rapidly disperses or packs together, this feature changes rapidly.

---

### Feature 5–7: Optical Flow Features (Motion Analysis)

To compute optical flow, consecutive frames are compared:

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray,
    None,
    pyr_scale=0.5,   # pyramid scale
    levels=3,        # pyramid levels
    winsize=15,      # smoothing window
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)
```

The result is a (H, W, 2) array where each pixel has (dx, dy) — how many pixels it moved horizontally and vertically.

From this flow field, three features are extracted:

**Feature 5: Mean Flow Magnitude**
```python
magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
mean_magnitude = np.mean(magnitude)
```
The average speed of all pixels. High mean magnitude = crowd moving fast overall. This is the most direct speed measurement. In normal walking, this is low and steady. During running or panic, it spikes dramatically.

**Feature 6: Variance of Flow Magnitude**
```python
var_magnitude = np.var(magnitude)
```
How uneven the motion is across the frame. A uniform crowd moving in one direction has low variance. A chaotic scene (some running, some stationary, some moving in different directions) has high variance. This is a key discriminator between orderly and disorderly crowd motion.

**Feature 7: Maximum Flow Magnitude**
```python
max_magnitude = np.max(magnitude)
```
The single fastest-moving pixel or region. This catches fast-moving individuals (runner, person fleeing) even if the overall crowd is slow. An anomaly can start with one person running before spreading.

---

### Feature 8: Directional Entropy (Motion Disorder)

```python
angles = np.arctan2(flow[..., 1], flow[..., 0])  # angle of each flow vector
hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi), density=True)
entropy = -np.sum(hist * np.log(hist + 1e-8))
```

The flow angles are computed for every pixel (which direction is it moving?). A histogram of 16 angular bins captures the distribution of motion directions. Shannon entropy measures how spread out this distribution is.

**Interpretation:**
- **Low entropy:** All pixels moving in roughly the same direction (orderly flow — pedestrians walking together)
- **High entropy:** Pixels moving in all different directions (chaotic motion — panic, fighting, dispersal)

Directional entropy is one of the most powerful features for detecting crowd panic, because panic characteristically causes people to move in all directions simultaneously rather than following normal orderly flow.

**Typical values:**
- Normal walking crowd: entropy ~1.5–2.0 (moderate order)
- Panic / chaos: entropy ~2.5–3.5 (near maximum disorder)

---

### Feature 9: Divergence Proxy (Crowd Expansion / Contraction)

```python
dx = flow[..., 0]  # horizontal flow
dy = flow[..., 1]  # vertical flow
div = np.abs(np.gradient(dx, axis=1) + np.gradient(dy, axis=0))
divergence_proxy = np.mean(div)
```

Divergence of the flow field measures whether the crowd is expanding (moving outward from a centre) or contracting (converging). High divergence indicates dispersal — people fleeing outward from a point, which is characteristic of panic events.

**Why it works:** When something frightening happens in the centre of a crowd, people move away from it radially. This produces high positive divergence around the event origin. Normal pedestrian flow has near-zero divergence.

---

### Feature 10: Temporal Contrast Change

```python
contrast_change = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
```

The mean absolute difference between the current frame and the previous frame in pixel values. This is a global motion energy measure — the larger it is, the more the scene has changed. Fast, large movements produce high values. This is complementary to optical flow: it captures any pixel-level change, not just structured motion.

---

## Stage 2: Temporal Window Aggregation

### Why Aggregation Is Needed

A single frame's 10 features are noisy and ambiguous. A person jogging looks briefly similar to panic on a single frame. The key signal is *how the features evolve over time*: does speed increase? Does entropy rise? Is the frame changing faster and faster?

### The Window

30 consecutive frames are collected (approximately 1.2 seconds at 25 fps). This creates a 30 × 10 matrix of features.

### Four Aggregations → 40 Dimensions

**Mean (10 values):** Average value of each feature across all 30 frames. Captures the typical state during the window. A window where the mean magnitude is high means the crowd was consistently fast throughout.

**Standard Deviation (10 values):** How much each feature varied within the window. High std means fluctuating, unstable motion. A calm crowd has low std on all features.

**Maximum (10 values):** The peak value of each feature during the window. Captures brief but intense spikes that the mean might smooth over. A single moment of extreme motion (one person running through the scene) will show in the maximum even if the mean is low.

**Delta — last third minus first third (10 values):**
```python
third = len(frames) // 3
delta = frames[-third:].mean(axis=0) - frames[:third].mean(axis=0)
```
The change in mean features from the beginning to the end of the window. A positive delta on mean magnitude means the crowd sped up during this window (acceleration). This is the most powerful feature for detecting *escalation* — a crowd that starts normal and becomes dangerous will show large positive deltas, especially on mean_magnitude, variance, and directional_entropy.

**The panic signature:** A typical panic event shows:
- Delta mean_magnitude: large positive (crowd accelerating)
- Delta directional_entropy: large positive (flow becoming chaotic)
- Maximum max_magnitude: very high (someone running)
- Std mean_magnitude: high (erratic speed changes)

---

## Stage 3: Random Forest Classification

### What is a Random Forest?

A Random Forest is an ensemble of decision trees. Each tree is trained on a random subset of the training data and a random subset of features at each split. The 300 trees independently vote on whether a window is anomalous, and the final probability is the fraction of trees that vote "anomalous."

```
Input: 40-dimensional feature vector
         │
         ├──► Tree 1 → 0 (normal)
         ├──► Tree 2 → 1 (anomalous)
         ├──► Tree 3 → 1 (anomalous)
         │    ...
         └──► Tree 300 → 1 (anomalous)

         Votes: 210 anomalous / 300 total
         Probability: 0.70 → MEDIUM alert
```

### Why 300 Trees?

More trees reduce variance (the model is less sensitive to random fluctuations in training data). Beyond ~200 trees, improvements diminish. 300 was chosen as a good balance of accuracy and inference speed.

### Class Balancing

Normal crowd footage is much more common than anomalous footage. Without correction, the model would learn to predict "normal" for everything (achieving high accuracy but missing all anomalies). Two techniques address this:

1. **`class_weight='balanced_subsample'`** — each tree automatically up-weights the minority class (anomalous) when computing splits, ensuring anomalies have equal influence in training.

2. **Downsampling** — the normal training dataset is reduced to 2× the number of anomalous samples during training data construction.

### Optimal Threshold

The Random Forest outputs a probability (0.0–1.0). By default, 0.5 is used as the decision boundary, but the actual optimal threshold (maximising F1 score) was found to be **0.537** via a threshold sweep on the validation set. This threshold is used in evaluation; in the demo, the thresholds are applied to the raw probability for the LOW/MEDIUM/HIGH alert system.

### Training Data

| Source | Samples | Label |
|--------|---------|-------|
| ShanghaiTech test clips (80%) | Anomalous and normal windows | Per ground truth mask |
| ShanghaiTech training videos (330) | Normal windows (subsampled) | 0 (all normal) |
| Total training windows | ~8,000+ | Mixed |
| Test set windows | 1,044 | Held out |

---

## Stage 4: Risk Score Fusion

The raw model probability is combined with three additional signals for a more robust final risk score:

```python
risk = (w1 * anomaly_score
      + w2 * flow_instability
      + w3 * density_pressure
      + w4 * trend_acceleration)
```

| Signal | Weight | Source | What it measures |
|--------|--------|--------|-----------------|
| anomaly_score | 0.40 | Random Forest output | Overall window anomaly probability |
| flow_instability | 0.25 | mean(variance_magnitude) across window | How chaotic the motion is |
| density_pressure | 0.20 | mean(occupancy) across window | How crowded the scene is |
| trend_acceleration | 0.15 | delta of mean_magnitude | Is the crowd speeding up? |

This multi-signal fusion means even if the Random Forest score is moderate, the system can still raise a HIGH alert if the optical flow is simultaneously extremely chaotic and the crowd density is very high.

---

## Stage 5: Exponential Smoothing

```python
smoothed = alpha * current_score + (1 - alpha) * previous_smoothed
# alpha = 0.35
```

Raw scores frame-to-frame are noisy. Smoothing with alpha=0.35 means the current score contributes 35% and recent history contributes 65%. This eliminates brief spikes from motion blur, lighting changes, or compression artefacts, while still responding quickly to real escalating events.

**Effect:** A single anomalous window will not trigger an immediate MEDIUM alert. Two or three consecutive anomalous windows in a row will cause the smoothed score to rise steadily and cross the threshold, creating a more reliable alert.

---

## Stage 6: Hysteresis

```python
# Transition up: score must exceed threshold + margin
# Transition down: score must fall below threshold - margin
margin = 0.05
```

Without hysteresis, if the smoothed score fluctuates around 0.50, the alert would switch between LOW and MEDIUM constantly. With hysteresis, the score must reach 0.55 to trigger MEDIUM, but once at MEDIUM it only reverts to LOW if the score falls below 0.45. This stabilises alerts.

---

## Feature Importances

Based on model inspection, the top features by importance are:

| Rank | Feature | Importance | Reason |
|------|---------|-----------|--------|
| 1 | Mean of occupancy (delta) | High | Crowd density changes signal dispersal |
| 2 | Mean of Laplacian variance | High | Edge energy captures motion intensity |
| 3 | Std of mean_magnitude | High | Erratic speed = chaos |
| 4 | Max of max_magnitude | High | Peak speed catches fleeing individuals |
| 5 | Mean of directional_entropy | High | Disorder is the strongest single signal |
| 6 | Delta of mean_magnitude | Medium | Acceleration = escalation |
| 7 | Mean of var_magnitude | Medium | Uneven motion = heterogeneous crowd behaviour |
| 8 | Max of directional_entropy | Medium | Worst-case chaos in window |
| 9 | Std of occupancy | Medium | Fluctuating density |
| 10 | Mean of contrast_change | Medium | Scene change rate |

---

## How Specific Anomaly Types Are Detected

### Running
- Mean flow magnitude rises sharply (fast motion)
- Directional entropy stays low-to-medium (everyone moving same direction)
- Delta mean_magnitude is large positive (acceleration)
- Max magnitude very high

### Fighting / Violent Altercation
- High variance flow magnitude (one small area very fast, rest normal)
- High Laplacian variance (rapid pixel changes)
- Directional entropy moderate-to-high (chaotic local motion)
- Scene stays localised — divergence low (not spreading)

### Panic / Mass Dispersal
- Directional entropy shoots to maximum (everyone in different directions)
- Divergence proxy high (outward expansion)
- Delta mean_magnitude large positive (escalating)
- Occupancy may drop rapidly (people fleeing out of frame)

### Dangerous Congestion
- Occupancy very high and rising
- Mean flow magnitude very low (crowd barely moving — gridlocked)
- Std of flow magnitude high (some moving, some stuck)

### Normal Walking (correctly NOT flagged)
- Low mean magnitude (slow, orderly pace)
- Low directional entropy (everyone going same direction)
- Near-zero delta (constant speed)
- Low variance (smooth, consistent motion)

---

## Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `artifacts/models/shanghaitech_windowed_rf.joblib` | 5.3 MB | Production model |
| `artifacts/models/shanghaitech_windowed_gbt.joblib` | 482 KB | GBT comparison |
| `artifacts/models/shanghaitech_ablation_noflow.joblib` | 6.2 MB | No optical flow variant |
| `artifacts/models/shanghaitech_ablation_w15.joblib` | 7.3 MB | W=15 window variant |
| `artifacts/reports/shanghaitech_windowed_rf_metrics.json` | — | Full evaluation metrics |
| `artifacts/reports/shanghaitech_windowed_rf_predictions.csv` | — | Per-window predictions |

---

## Complete Performance Report

### Overall Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8313 |
| PR-AUC | 0.8261 |
| F1 (threshold 0.537) | 0.787 |
| Recall (anomaly) | 0.885 |
| Precision (anomaly) | 0.709 |
| Accuracy | 0.761 |
| Test set size | 1044 windows |

### Ablation Study Results

| Variant | ROC-AUC | What was changed |
|---------|---------|-----------------|
| RF W=30 (production) | **0.8313** | Baseline |
| GBT W=30 | 0.8178 | Classifier changed to Gradient Boosting |
| RF W=30, no flow | 0.8016 | Optical flow features zeroed |
| RF W=15 | 0.7449 | Window size halved |

**Key findings:**
1. Window size is the most critical factor (−8.6 AUC points when halved)
2. Optical flow contributes ~3 AUC points
3. RF slightly outperforms GBT on this dataset
4. Even without optical flow, the model achieves 0.80 AUC (intensity and temporal features still carry the load)

---

## Limitations and Failure Modes

### Known Weak Scenes
- **Scenes 07 and 08:** These scenes have an *inverse* anomaly signature — the anomaly is very few people (unusual for a normally busy scene). The model is trained to detect chaotic/fast motion, so scenes where "anomaly = empty" are harder to detect.

### General Limitations
- **Camera angle dependency:** The model was trained on elevated surveillance angles. Frontal or very low-angle cameras produce different optical flow patterns and may reduce accuracy.
- **Lighting changes:** Sudden lighting changes (clouds, artificial lights switching) can temporarily raise the Laplacian variance and contrast features, producing brief false positives.
- **Single person anomalies:** If only one person is acting anomalously in a large crowd, the window-level features may average it out. The max features partially compensate for this.
- **Trained on campus data only:** The ShanghaiTech dataset is from a university campus. Behaviour norms differ in stadiums, train platforms, or dense markets. Performance on very different environments may be lower until the model is retrained on domain-specific data.
