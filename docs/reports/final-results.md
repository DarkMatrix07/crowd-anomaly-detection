# Final Results Summary

## Current Build Status

- Pipeline status: end-to-end scaffolding complete through API and dashboard workflow.
- Test status: automated unit/integration coverage active for data, features, training, risk, inference, and API.
- Dataset status: ShanghaiTech ingestion/training pending (not yet executed in this repository).

## Latest Verified Commands

```bash
python -m pytest -q
python scripts/train_baseline.py --config configs/train.yaml
python scripts/evaluate_model.py --ckpt dummy --out docs/reports/baseline-threshold-sweep.csv
python scripts/run_inference.py --max-frames 20
python scripts/run_ablations.py --out-csv docs/reports/ablation-results.csv --out-md docs/reports/ablation-study.md
```

## Operator Stack

- API: `uvicorn src.api.app:app --reload --port 8000`
- Dashboard: `streamlit run dashboard/app.py`
- Supported operator controls:
  - camera filtering in dashboard summary
  - threshold profile updates via `/config/thresholds`
  - alert acknowledgment notes via `/alerts/{id}/ack`

## Metrics (Current Placeholder Run)

- Baseline metrics and ablations currently reflect synthetic/demo inputs.
- Real project acceptance metrics (ROC-AUC, event F1, lead-time) remain pending until ShanghaiTech training/evaluation.

## Required Before Supervisor Final Review

1. Download and structure ShanghaiTech subset (`part_a` primary).
2. Run real training/evaluation and update `baseline-eval.md`.
3. Re-run ablations with real model outputs and update `ablation-study.md`.
4. Record flow visualizations and live demo evidence.
