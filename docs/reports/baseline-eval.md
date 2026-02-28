# Baseline Evaluation Report

## Model and Data Context

- Model checkpoint: `artifacts/models/shanghaitech_windowed_rf.joblib`
- Classifier: RandomForestClassifier (300 estimators, balanced_subsample class weight)
- Approach: Temporal-window aggregation (W=30 frames, stride=5 training / stride=1 test)
- Feature dimensionality: 40-d per window (mean + std + max + delta of 10 frame features)
- Dataset: ShanghaiTech Campus dataset — testing split (107 clips, 60 scenes)
- Evaluation split: clip-level stratified 80/20 train/test (85 train clips, 22 test clips)
  - Stratification by scene prefix prevents data leakage across scenes
- Frame preprocessing: resized to 320×240 before feature extraction; frame stride = 5

## Core Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8313** |
| PR-AUC | 0.8261 |
| F1 (optimal threshold 0.537) | 0.787 |
| F1 (default threshold 0.5) | 0.784 |
| Accuracy | 0.761 |
| Precision | 0.709 |
| Recall | 0.885 |

**Confusion matrix** (test set, 1044 windows):

|  | Predicted Normal | Predicted Anomaly |
|--|--|--|
| **Actual Normal** | TN = 332 | FP = 190 |
| **Actual Anomaly** | FN = 60 | TP = 462 |

## Scene-Level ROC-AUC

| Scene | AUC | Test Windows | Notes |
|-------|-----|-------------|-------|
| 01 | 0.693 | 282 | Mixed — multi-person corridor |
| 02 | 0.877 | 63 | Good — open plaza |
| 03 | 0.849 | 83 | Good — staircase events |
| 04 | 0.875 | 188 | Good — outdoor crowd |
| 05 | **1.000** | 126 | Perfect — distinctive panic events |
| 06 | — | — | Single class in test split |
| 07 | 0.481 | 136 | Weak — visually ambiguous scene |
| 08 | 0.473 | 54 | Weak — small anomaly signatures |
| 10 | — | — | Single class in test split |
| 12 | 0.849 | 39 | Good |

## Threshold Selection

- Operating threshold selected by F1 sweep over ROC curve: **0.537**
- At this threshold: Recall = 0.885 (catches 88.5% of anomalous windows)
- False positive rate at optimal threshold: 190 / 522 normal windows = 36%
- For deployment, threshold can be raised to ~0.65 to reduce false positives at the cost of some recall

## Top Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | mean_occupancy | 0.055 |
| 2 | max_occupancy | 0.053 |
| 3 | mean_lap_var | 0.035 |
| 4 | std_occupancy | 0.033 |
| 5 | mean_flow_max | 0.032 |
| 6 | mean_mean_intensity | 0.032 |
| 7 | max_mean_intensity | 0.031 |
| 8 | mean_std_intensity | 0.029 |
| 9 | mean_temporal_contrast | 0.028 |
| 10 | std_temporal_contrast | 0.028 |

Occupancy (fraction of non-background pixels) is the strongest single discriminator, followed by edge sharpness (Laplacian variance) and motion magnitude (flow_max).

## Failure Analysis

- **Scene 07 / 08 weakness**: AUC below 0.5 indicates the model's occupancy/intensity signals are inversely correlated with ground truth labels in these scenes. Likely cause: anomaly events in these scenes involve *fewer* people (e.g., a single person running) rather than crowd-level panic, making density-based features unreliable.
- **False positives**: Primarily in dense-crowd normal windows that have high occupancy — the model over-triggers on crowded but normal scenes.
- **Low-light / camera motion**: Not explicitly handled; temporal contrast and flow features will inflate in scenes with illumination changes. No night-time clips in this dataset.

## Supervisor Checkpoint

- Review status: Pending
- Review date:
- Reviewer comments:
