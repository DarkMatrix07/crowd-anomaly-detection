# Abnormal Crowd Behaviour Detection

Implementation scaffold for abnormal crowd behaviour detection and early warning.

## Prerequisites

- Python 3.11+

## Setup

```bash
python -m pip install -r requirements.txt
```

## CLI Usage

```bash
python -m src --help
python -m src test
python -m src train --config configs/train.yaml
python -m src infer --source path/to/video.mp4 --config configs/infer.yaml
```

## Service + Dashboard

```bash
uvicorn src.api.app:app --reload --port 8000
streamlit run dashboard/app.py
```

## Evaluation + Ablations

```bash
python scripts/evaluate_model.py --ckpt <path-to-checkpoint> --out docs/reports/baseline-threshold-sweep.csv
python scripts/run_ablations.py --out-csv docs/reports/ablation-results.csv --out-md docs/reports/ablation-study.md
```

## Notes

- `train` and `infer` currently run scaffold pipelines; replace synthetic sources with ShanghaiTech data for real metrics.
- `ablation-study.md` is currently generated from synthetic scores until full dataset training is run.
