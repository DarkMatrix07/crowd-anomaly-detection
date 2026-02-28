# Ablation Study

All variants trained on ShanghaiTech Campus dataset with clip-level 80/20 stratified split.
Evaluation is window-level (stride=1 on test clips). Training uses frame-stride=5, resize=320×240.

## Results

| Variant | ROC-AUC | PR-AUC | F1 | Precision | Recall | Notes |
|---------|---------|--------|----|-----------|--------|-------|
| **RF W=30 (production)** | **0.8313** | **0.8261** | **0.787** | 0.709 | 0.885 | Baseline production model |
| GBT W=30 | 0.8178 | 0.8020 | 0.765 | 0.668 | 0.897 | GradientBoosting, lr=0.05, depth=4 |
| RF W=30 no-flow | 0.8016 | 0.7660 | 0.786 | 0.680 | 0.931 | Optical flow features zeroed out |
| RF W=15 | 0.7449 | 0.6711 | 0.715 | 0.604 | 0.876 | Shorter temporal context |

## Findings

### 1. Window size matters (W=30 vs W=15)
Reducing the window from 30 to 15 frames drops ROC-AUC by **8.6 points** (0.831 → 0.745).
Anomalies in ShanghaiTech are multi-second events; a 30-frame window (~1.2 s at 25 fps) captures enough temporal evolution for the delta features to activate, while 15 frames is too short.

### 2. Optical flow adds measurable signal (RF full vs RF no-flow)
Removing all flow-derived features (flow_mean, flow_var, flow_max, directional_entropy, divergence_proxy and their window aggregates) drops ROC-AUC by **3.0 points** (0.831 → 0.802). The degradation is modest because static features (occupancy, Laplacian variance, intensity) carry most of the signal at this scale. Flow features are more important in scenes with subtle motion anomalies.

### 3. RandomForest vs GradientBoosting
RF slightly outperforms GBT on this dataset (0.831 vs 0.818 AUC). Both are strong. GBT achieves slightly higher recall (0.897 vs 0.885) but at the cost of more false positives. RF is selected as the production model due to higher overall AUC and simpler calibration.

### 4. Frame-level baseline (previous iteration, no windowing)
Earlier experiment with pure frame-level RF (no temporal aggregation) achieved ROC-AUC = 0.6623 — **17 points below** the W=30 windowed model. This confirms that temporal context is the critical factor: single frames lack the temporal patterns (acceleration, escalation) that distinguish anomalies from normal dense crowds.

## Recommended Production Configuration

```
Classifier:    RandomForest (300 estimators, balanced_subsample)
Window size:   30 frames
Window stride: 5 frames (training), 1 frame (inference)
Frame stride:  5 (feature extraction)
Resize:        320 × 240
Threshold:     0.537 (F1-optimal) or 0.650 (low false-alert deployment)
```

Artifact: `artifacts/models/shanghaitech_windowed_rf.joblib`
