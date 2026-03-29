# Final Results Summary

## Project Status: COMPLETE

All models trained, evaluated, and integrated. Demo ready.

---

## Model Comparison

| Model | Approach | ROC-AUC | PR-AUC | Accuracy | F1 |
|-------|----------|---------|--------|----------|----|
| **ResNet18 + MLP** | Transfer learning (DL) | **0.9715** | **0.9515** | **91.96%** | 0.92 |
| Random Forest (W=30) | Hand-crafted features + RF | 0.8313 | 0.8261 | — | 0.787 |
| CNN Autoencoder | Unsupervised reconstruction | 0.5483 | 0.4490 | — | — |

**Production model: ResNet18 + MLP** (highest AUC, satisfies DL requirement)
**Baseline model: Random Forest W=30** (available for comparison in demo)

---

## ResNet18 + MLP (Primary)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9715** |
| PR-AUC | **0.9515** |
| Accuracy (threshold=0.65) | **91.96%** |
| Anomaly Recall | **99%** |
| Anomaly Precision | 85% |
| Normal Recall | 87% |
| Normal Precision | 99% |
| F1 macro avg | 0.92 |

Architecture: Frozen ResNet18 (ImageNet) → 512-d/frame → window aggregation (mean+std+max+delta) → 2048-d → MLP (2048→256→64→1)

## Random Forest W=30 (Baseline)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8313 |
| PR-AUC | 0.8261 |
| F1 (threshold=0.537) | 0.787 |
| Anomaly Recall | 88.5% |

## CNN Autoencoder (Ablation — Failed)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.5483 (≈ random) |
| PR-AUC | 0.4490 |

Conclusion: Unsupervised reconstruction fails on this dataset — the model reconstructs anomalous scenes as well as normal ones.

---

## Ablation Study (RF variants)

| Variant | ROC-AUC | Δ vs production |
|---------|---------|-----------------|
| RF W=30 (production) | 0.8313 | — |
| GBT W=30 | 0.8180 | −0.013 |
| RF W=30 no optical flow | 0.8020 | −0.029 |
| RF W=15 | 0.7450 | −0.086 |

---

## Model Artifacts

| File | Description |
|------|-------------|
| `artifacts/models/resnet_mlp.pt` | **Primary: ResNet18+MLP** |
| `artifacts/models/shanghaitech_windowed_rf.joblib` | Baseline: RF W=30 |
| `artifacts/models/autoencoder.pt` | Ablation: CNN Autoencoder |
| `artifacts/models/shanghaitech_windowed_gbt.joblib` | Ablation: GBT W=30 |
| `artifacts/models/shanghaitech_ablation_noflow.joblib` | Ablation: RF no-flow |
| `artifacts/models/shanghaitech_ablation_w15.joblib` | Ablation: RF W=15 |

---

## Key Commands

```bash
# Train ResNet+MLP (primary DL model)
python scripts/train_resnet_mlp.py

# Train RF baseline
python scripts/train_production_model.py

# Train autoencoder
python scripts/train_autoencoder.py

# Evaluate ResNet+MLP
python scripts/evaluate_resnet_mlp.py

# Evaluate autoencoder
python scripts/evaluate_autoencoder.py

# Run demo (model selector included)
python -m streamlit run scripts/crowd_anomaly_demo.py

# API server
uvicorn src.api.app:app --reload --port 8000

# Dashboard
streamlit run dashboard/app.py
```

---

## Checklist

- [x] ResNet18+MLP trained — ROC-AUC 0.9715
- [x] RF baseline trained — ROC-AUC 0.8313
- [x] CNN Autoencoder trained — documented as failed baseline
- [x] Ablation study complete (4 variants)
- [x] Model selector in live demo
- [x] FastAPI + Streamlit dashboard operational
- [x] Reports updated
- [ ] Supervisor final review
