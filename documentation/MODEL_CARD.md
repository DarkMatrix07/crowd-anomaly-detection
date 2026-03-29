# Model Card — ResNet18 + MLP Anomaly Detector

---

## Model Details

| Field | Value |
|-------|-------|
| Name | ResNet18 + WindowMLP |
| Version | 1.0 |
| Type | Binary classifier (normal vs anomaly) |
| Architecture | Frozen ResNet18 (feature extractor) + MLP (classifier) |
| Framework | PyTorch 2.x |
| Artifact | `artifacts/models/resnet_mlp.pt` |
| Developed by | Capstone Team, SRM University AP |
| Date | 2026 |

---

## Intended Use

**Primary use:** Detecting abnormal crowd behaviour in surveillance video footage from fixed campus cameras.

**Intended users:** Security operators monitoring campus surveillance systems.

**Deployment context:** Offline video analysis and live clip-based demo. Not intended for real-time streaming without additional infrastructure.

---

## Model Architecture

```
Input: Video clip (N frames, BGR, any resolution)
  │
  ▼  Preprocessing: resize to 224×224, normalise (ImageNet mean/std)
  │
  ▼  ResNet18 backbone (frozen, ImageNet pretrained)
     → 512-dimensional feature vector per frame
  │
  ▼  Temporal window aggregation (W=30 frames)
     mean + std + max + delta across 512 dims
     → 2048-dimensional window vector
  │
  ▼  WindowMLP:
     Linear(2048→256) → BatchNorm → ReLU → Dropout(0.3)
     Linear(256→64)   → BatchNorm → ReLU → Dropout(0.3)
     Linear(64→1)     → Sigmoid
  │
  ▼  Output: anomaly score ∈ [0, 1]
```

---

## Training Data

**Dataset:** ShanghaiTech Campus Dataset

| Split | Source | Content |
|-------|--------|---------|
| Normal (train) | `training/videos/` | ~330 normal .avi clips from 13 campus scenes |
| Anomaly (train) | `testing/frames/` (80% of clips) | Anomalous clips with frame-level ground truth masks |
| Test | `testing/frames/` (20% of clips) | Held-out clips, never seen during training |

**Split strategy:** Clip-level 80/20 split (not frame-level) to prevent data leakage between similar frames within the same clip.

**Label source:** Frame-level binary masks manually annotated by the ShanghaiTech dataset authors.

**Class imbalance handling:** `pos_weight` in Binary Cross Entropy loss, set proportional to the ratio of normal to anomaly frames.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 20 |
| Window size | 30 frames |
| Stride (inference) | 10 frames |
| Dropout | 0.3 |
| Loss | Binary Cross Entropy with pos_weight |

---

## Performance

Evaluated on the held-out 20% test split of ShanghaiTech.

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9715** |
| PR-AUC | **0.9515** |
| Accuracy (threshold=0.65) | **91.96%** |
| Anomaly Recall | 99% |
| Anomaly Precision | 85% |
| Normal Recall | 87% |
| Normal Precision | 99% |
| F1 (macro) | 0.92 |

**Comparison with baseline:**

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| ResNet18 + MLP (this model) | 0.9715 | Production model |
| Random Forest W=30 | 0.8313 | ML baseline |

---

## Limitations

**1. Dataset-specific generalisation**
The model was trained and evaluated on a single dataset (ShanghaiTech Campus). Performance on footage from different cameras, locations, or lighting conditions may be lower. Fine-tuning on target-domain data is recommended for deployment in a new environment.

**2. Fixed temporal window**
The model processes 30-frame windows. Very short anomalies (<1 second) may be diluted across a window and missed. Very long anomalies are detected but scored per-window, not globally.

**3. No crowd density awareness**
The model does not explicitly model crowd density. A small number of people running in an otherwise empty scene can trigger the same motion features as a crowd anomaly.

**4. CPU inference speed**
On CPU, processing a 457-frame clip takes approximately 45–60 seconds. Real-time deployment at 30fps requires GPU inference.

**5. No live stream support**
Currently operates on pre-recorded frame sequences. A frame capture loop is required for live RTSP/IP camera input.

**6. Fixed camera angle**
Designed for fixed surveillance cameras. PTZ (pan-tilt-zoom) cameras or moving cameras would produce unstable optical flow that could trigger false positives.

---

## Ethical Considerations

**Privacy:** The model processes visual data from public spaces. Deployment should comply with applicable data protection regulations and institutional policies. Video data should not be retained beyond operational requirements.

**Bias:** The training data covers a Chinese university campus. Performance may differ on populations, clothing styles, or crowd behaviours not represented in the training set.

**Human oversight:** This model is designed as a decision support tool, not a decision-making system. All alerts require human review before any action is taken. No automated consequences (alarms, access control, law enforcement notification) should be triggered directly by model output.

**False positives:** An inherent risk in anomaly detection. A false positive rate of ~13% (1 - 87% normal recall) means approximately 1 in 8 normal scenes may be flagged. Threshold tuning can reduce this at the cost of reduced anomaly recall.

**Transparency:** This model card documents the model's capabilities, limitations, and ethical considerations in line with responsible AI practices.

---

## Random Forest Baseline Model Card

| Field | Value |
|-------|-------|
| Name | Random Forest W=30 |
| Type | Binary classifier |
| Features | Optical flow + HOG, 30-frame window, 40-d |
| Artifact | `artifacts/models/shanghaitech_windowed_rf.joblib` |
| ROC-AUC | 0.8313 |
| Inference time (CPU) | ~2–5 seconds per clip |
| Intended use | Comparison baseline, lightweight fallback |
| Limitation | No deep feature learning, lower accuracy |
