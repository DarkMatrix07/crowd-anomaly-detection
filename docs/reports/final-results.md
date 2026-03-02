# Final Results Summary

## Current Build Status

- Pipeline status: fully integrated — trained model wired into `RollingInferencePipeline` via `src/inference/anomaly_model.py`
- Model status: trained and evaluated on real ShanghaiTech Campus data (ROC-AUC 0.831)
- Test status: automated unit/integration coverage active for data, features, training, risk, inference, and API

## Model Performance

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| ROC-AUC | **0.8313** | ≥ 0.75 | ✓ PASSED |
| PR-AUC | 0.8261 | — | — |
| F1 (optimal threshold) | 0.787 | — | — |
| Recall (anomaly detection) | 0.885 | — | — |

Full details: [`docs/reports/baseline-eval.md`](baseline-eval.md)
Ablation study: [`docs/reports/ablation-study.md`](ablation-study.md)

## Key Training Commands

```bash
# Production model (RF, W=30)
python scripts/train_production_model.py \
  --test-ratio 0.20 --window-stride 5 --frame-stride-clips 5 \
  --max-train-videos 0 --n-estimators 300 --classifier rf \
  --model-out artifacts/models/shanghaitech_windowed_rf.joblib \
  --metrics-out artifacts/reports/shanghaitech_windowed_rf_metrics.json

# Load model in Python
from src.inference.anomaly_model import load_anomaly_model_fn
anomaly_fn = load_anomaly_model_fn()

# Wire into pipeline
from src.inference.pipeline import RollingInferencePipeline
pipe = RollingInferencePipeline(
    camera_id="cam_01", clip_length=30, clip_stride=10,
    anomaly_model_fn=anomaly_fn,
)
```

## Operator Stack

- API: `uvicorn src.api.app:app --reload --port 8000`
- Dashboard: `streamlit run dashboard/app.py`
- Supported operator controls:
  - Camera filtering in dashboard summary
  - Threshold profile updates via `/config/thresholds`
  - Alert acknowledgment notes via `/alerts/{id}/ack`

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/models/shanghaitech_windowed_rf.joblib` | Production RF model (W=30) |
| `artifacts/models/shanghaitech_windowed_gbt.joblib` | GBT variant (W=30) |
| `artifacts/models/shanghaitech_ablation_noflow.joblib` | Ablation: no optical flow |
| `artifacts/models/shanghaitech_ablation_w15.joblib` | Ablation: W=15 window |
| `artifacts/reports/shanghaitech_windowed_rf_metrics.json` | Full RF metrics (JSON) |
| `artifacts/reports/shanghaitech_windowed_gbt_metrics.json` | GBT metrics (JSON) |
| `artifacts/reports/shanghaitech_ablation_noflow_metrics.json` | No-flow ablation metrics |
| `artifacts/reports/shanghaitech_ablation_w15_metrics.json` | W=15 ablation metrics |

## Required Before Supervisor Final Review

1. ✓ Train real model on ShanghaiTech (ROC-AUC 0.831 — gate cleared)
2. ✓ Update `baseline-eval.md` with real numbers
3. ✓ Run real ablations and update `ablation-study.md`
4. ✓ Wire trained model into `RollingInferencePipeline`
5. Record flow visualizations and live demo evidence for presentation
6. Supervisor review of `baseline-eval.md` and `ablation-study.md`
