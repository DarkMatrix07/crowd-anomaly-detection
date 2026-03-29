# Abnormal Crowd Behaviour Detection and Early Warning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and validate a video-based system that detects abnormal crowd behavior (panic movement, congestion, bottlenecks) and raises early warnings before critical escalation.

**Architecture:** The system uses an offline training pipeline and an online inference pipeline. Offline flow: dataset ingestion -> preprocessing/clip generation -> feature extraction (optical flow + density/motion stats) -> baseline anomaly model training -> risk-score calibration. Online flow: camera stream -> rolling clip inference -> risk scoring + temporal smoothing -> warning trigger and dashboard/API output.

**Tech Stack:** Python 3.11+, PyTorch, OpenCV, NumPy/Pandas, scikit-learn, TorchMetrics, Hydra/OmegaConf, FastAPI (serving), Streamlit or React dashboard, MLflow/WandB for experiment tracking, pytest for validation, Docker for reproducible deployment.

---

## 1) Starting Point (from the PDF)

- Current completion reported: **60%**.
- Already done:
- Problem statement and scope finalized.
- ShanghaiTech selected as primary dataset.
- High-level pipeline and DGX workflow planned.
- Literature review completed.
- Immediate pending work:
- Video preprocessing and clip generation.
- Baseline anomaly model training.
- Optical flow integration.
- Initial evaluation.
- Next-month targets:
- Risk scoring and early warning logic.
- Ablation studies.
- Preliminary results review.

## 2) Success Criteria (Definition of Done)

- Functional:
- System detects anomalous crowd behavior on validation/test videos.
- System produces warning levels (`LOW`, `MEDIUM`, `HIGH`) with timestamped evidence.
- Real-time inference supports at least one live or recorded camera stream.
- Quality:
- Frame-level ROC-AUC >= 0.80 on held-out data.
- Event-level F1 >= 0.70 for abnormal segments.
- Mean alert lead time >= 3 seconds before incident onset. If precise onset labels are unavailable, use a human-annotated onset-frame proxy on a 20-clip validation subset.
- False alert rate <= 1 per 10 minutes on normal-only footage.
- Engineering:
- Reproducible training run from a single command.
- Versioned configs, saved checkpoints, and experiment logs.
- Unit/integration tests for data pipeline, risk scoring, and API contracts.

## 3) Project Structure to Create

```text
project-root/
  configs/
    data.yaml
    train.yaml
    infer.yaml
    risk.yaml
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/
    data/
      ingest.py
      preprocess.py
      clips.py
    features/
      optical_flow.py
      crowd_metrics.py
    models/
      baseline.py
      losses.py
      train.py
      evaluate.py
    risk/
      score.py
      calibrate.py
      alerts.py
    inference/
      stream.py
      pipeline.py
    api/
      app.py
      schemas.py
  tests/
    data/
    features/
    models/
    risk/
    api/
  scripts/
    prepare_data.py
    train_baseline.py
    evaluate_model.py
    run_inference.py
  docs/
    plans/
    reports/
```

## 4) Detailed Execution Plan

### Task 1: Repository and Environment Baseline

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `README.md`
- Create: `src/__init__.py`, `tests/test_smoke.py`

**Steps:**
1. Add dependency and tooling config (PyTorch, OpenCV, pytest, lint/format).
2. Add a failing smoke test for package import.
3. Create minimal package scaffold so test passes.
4. Add `make` or script aliases for `test`, `train`, `infer`.
5. Verify with: `pytest -q` and `python -m src --help` (or equivalent entrypoint).

**Output:** Reproducible dev environment with passing base test suite.

### Task 2: Dataset Ingestion and Metadata Index

**Files:**
- Create: `src/data/ingest.py`, `scripts/prepare_data.py`
- Create: `tests/data/test_ingest.py`
- Create: `configs/data.yaml`
- Create: `data/interim/metadata.csv` (generated)

**Steps:**
1. Implement dataset path validation and split discovery.
2. Add dataset subset control in `configs/data.yaml`:
   - `dataset.name: shanghaitech`
   - `dataset.subset: part_a` (default), options: `part_a | part_b | both`
3. Parse video/annotation files into a metadata table (`video_id`, `split`, `fps`, `frames`, label availability, `subset`).
4. Write tests for missing files, invalid paths, row-count consistency, and subset filtering.
5. Run: `python scripts/prepare_data.py --stage ingest`.
6. Verify generated metadata row counts against dataset folders.

**Output:** Reliable index of all train/val/test assets.

### Task 3: Video Preprocessing and Clip Generation

**Files:**
- Create: `src/data/preprocess.py`, `src/data/clips.py`
- Create: `tests/data/test_preprocess.py`, `tests/data/test_clips.py`
- Generate: `data/processed/clips/*.mp4` or tensor files

**Steps:**
1. Define canonical data hyperparameters in `configs/data.yaml`: `target_fps`, `frame_size`, `clip_length_frames`, `clip_stride_frames`, `num_workers`, `prefetch_factor`.
2. Implement transform pipeline (decode -> resize -> normalize -> clip windows).
3. Add augmentation pipeline for generalization:
   - spatial: random crop, horizontal flip, brightness/contrast jitter
   - temporal: FPS jitter / clip speed perturbation
4. Add tests for clip count, shape, boundary handling, and deterministic no-augment mode.
5. Run preprocessing on a small sample split.
6. Run full preprocessing on DGX and log throughput with tuned dataloader workers and batch size profile.

**Output:** Model-ready clips with deterministic preprocessing config.

### Task 4: Optical Flow and Crowd Motion Feature Extraction

**Files:**
- Create: `src/features/optical_flow.py`, `src/features/crowd_metrics.py`
- Create: `tests/features/test_optical_flow.py`, `tests/features/test_crowd_metrics.py`
- Create: `docs/reports/flow-visualization.md`

**Steps:**
1. Implement optical flow extraction with RAFT as primary (DGX path) and Farneback as CPU fallback.
2. Compute motion summaries: mean flow magnitude, variance, divergence proxy, directional entropy.
3. Add density proxy features (foreground occupancy, local crowd concentration index).
4. Validate feature stability on normal vs abnormal clips.
5. Save representative flow visualizations for normal/anomalous scenes for supervisor/demo reporting.
6. Persist features to `data/processed/features.parquet`.

**Output:** Reusable spatio-temporal feature set for training and risk scoring.

### Task 5: Baseline Anomaly Model Training

**Files:**
- Create: `src/models/baseline.py`, `src/models/losses.py`, `src/models/train.py`
- Create: `scripts/train_baseline.py`
- Create: `tests/models/test_baseline_shapes.py`
- Update: `configs/train.yaml`

**Steps:**
1. Choose baseline architecture (recommended: ConvLSTM autoencoder or 3D CNN binary classifier depending on label granularity).
2. Implement dataloaders for clip tensors + optional flow channels.
3. Add DGX-first training loop with DistributedDataParallel (`torchrun`), `torch.cuda.amp` mixed precision, checkpointing, and early stopping.
4. Add DGX tuning keys in config (`global_batch_size`, `per_gpu_batch_size`, `num_workers`, `grad_accum_steps`, `pin_memory`).
5. Track metrics each epoch (loss, AUC, PR-AUC).
6. Run baseline training on DGX and save best checkpoint.

**Output:** First trained model with reproducible config and logs.

### Task 6: Evaluation Harness and Error Analysis

**Files:**
- Create: `src/models/evaluate.py`, `scripts/evaluate_model.py`
- Create: `tests/models/test_evaluate_metrics.py`
- Create: `docs/reports/baseline-eval.md`

**Steps:**
1. Implement frame-level and clip-level scoring.
2. Generate ROC, PR curves, confusion matrix, and threshold sweep table.
3. Add event-level evaluation (segment overlap / temporal detection).
4. Produce failure-case report by scene type (dense, low light, camera shake).
5. Freeze baseline metrics as reference for ablation comparison.
6. Supervisor checkpoint: review `docs/reports/baseline-eval.md` before Task 7 starts.

**Output:** Quantified baseline performance and key failure patterns.

### Gate After Task 6 (Mandatory Iteration Loop)

- If frame-level ROC-AUC < 0.75, loop back to Task 5 for architecture/training adjustments before starting Task 7.
- If ROC-AUC >= 0.75, proceed to Task 7 risk calibration.

### Task 7: Risk Scoring and Early Warning Logic

**Files:**
- Create: `src/risk/score.py`, `src/risk/calibrate.py`, `src/risk/alerts.py`
- Create: `tests/risk/test_score.py`, `tests/risk/test_alerts.py`
- Create: `configs/risk.yaml`

**Steps:**
1. Define composite risk score:
   `risk = w1*anomaly_score + w2*flow_instability + w3*density_pressure + w4*trend_acceleration`.
2. Calibrate weights and thresholds on validation data (grid/Bayesian search).
3. Add temporal smoothing and hysteresis to reduce alert flicker.
4. Define alert policy:
   `LOW` (watch), `MEDIUM` (operator check), `HIGH` (immediate intervention).
5. Validate false alert rate and lead-time performance against success criteria.

**Output:** Stable warning mechanism beyond raw anomaly prediction.

### Task 8: Real-Time Inference Pipeline (Traffic/Surveillance Feed)

**Files:**
- Create: `src/inference/stream.py`, `src/inference/pipeline.py`, `scripts/run_inference.py`
- Create: `tests/inference/test_pipeline_contract.py`

**Steps:**
1. Implement RTSP/video file reader with reconnect and frame-drop handling.
2. Add rolling window buffer for clip-based inference.
3. Integrate model inference + risk scoring in a single pipeline.
4. Emit JSON events (`timestamp`, `camera_id`, `risk_level`, `score`, `evidence_window`).
5. Benchmark FPS and end-to-end latency under target hardware.

**Output:** Working near-real-time inference service.

### Task 9A: API and Alert Persistence Layer

**Files:**
- Create: `src/api/app.py`, `src/api/schemas.py`
- Create: `tests/api/test_alert_endpoint.py`

**Steps:**
1. Expose REST endpoints for current status, recent alerts, and health.
2. Add persistence for alert logs (SQLite/PostgreSQL).
3. Verify API schema and persistence flow with integration tests.

**Output:** Stable alert API service with stored event history.

### Task 9B: Monitoring Dashboard and Operator Workflow

**Files:**
- Create: `dashboard/` app (Streamlit or React)
- Create: `tests/api/test_dashboard_data_contract.py`

**Steps:**
1. Build dashboard views for live feed, risk timeline, and alert history.
2. Add operator controls (camera select, threshold profile, acknowledgment notes).
3. Validate end-to-end dashboard integration with API alerts and persisted records.

**Output:** Usable operator-facing early warning interface.

### Task 10: Ablations, Documentation, and Final Review Pack

**Files:**
- Create: `docs/reports/ablation-study.md`, `docs/reports/final-results.md`
- Update: `README.md`, `docs/plans/*`

**Steps:**
1. Run ablations:
   - without optical flow
   - without density features
   - without temporal smoothing
   - alternate model backbone
2. Compare metrics and highlight the best trade-off model.
3. Document reproducibility commands and deployment steps.
4. Prepare supervisor review deck: objective, method, results, limitations, next work.
5. Freeze release tag and archive artifacts.

**Output:** Complete technical package ready for academic and demo review.

## 5) Week-by-Week Timeline (8 Weeks)

- Week 1: Task 1-2 (environment + ingestion index)
- Week 2: Task 3 (preprocessing + clip generation)
- Week 3: Task 4 (optical flow + motion/density features)
- Week 4: Task 5 (baseline training)
- Week 5: Task 6 (evaluation + failure analysis)
- Week 6: Task 7 (risk scoring + early warning calibration)
- Week 7: Task 8 + Task 9A (real-time pipeline + API/persistence)
- Week 8: Task 9B + Task 10 (dashboard + ablations/final packaging)

## 6) Team Ownership Mapping (4 Members)

- Member A (Manas): Task 1-2 + Task 7 (core architecture, risk calibration, integration decisions).
- Member B (Vaibhav): Task 3-4 (preprocessing pipeline, optical flow and motion features).
- Member C (Nagendra): Task 5-6 (training/evaluation loops, DGX profiling, metric reporting).
- Member D (Srijan): Task 8-9B + Task 10 support (inference runtime, API/dashboard, demo packaging).
- Cross-check rule: each major task is reviewed by one non-owner before supervisor submission.

## 7) Verification Checklist Before Calling It Complete

- `pytest -q` passes for all modules.
- End-to-end run command executes without manual intervention:
  - `python scripts/prepare_data.py --stage all`
  - `python scripts/train_baseline.py --config configs/train.yaml`
  - `python scripts/evaluate_model.py --ckpt <best_model>`
  - `python scripts/run_inference.py --source <rtsp_or_video>`
- Baseline and final model metrics documented in `docs/reports/`.
- Risk threshold justification documented with validation curves.
- Demo scenario recorded with at least one normal and one abnormal case.

## 8) Risk Register and Mitigations

- Label quality mismatch with real crowd panic behavior:
- Mitigation: include weak supervision + manual validation subset.
- Domain shift from ShanghaiTech to target traffic/surveillance cameras:
- Mitigation: test-time adaptation, camera-specific normalization, and threshold calibration per site.
- High false positives in dense but normal crowd scenes:
- Mitigation: temporal smoothing, hysteresis, density-aware risk weighting.
- Real-time performance bottleneck:
- Mitigation: ONNX/TensorRT export, lower input resolution, batched clip inference.

## 9) Immediate Next Actions (This Week)

1. Finalize repository scaffold (Task 1).
2. Implement ingestion + metadata indexing (Task 2).
3. Start preprocessing pipeline on a sample subset (Task 3 partial).
4. Lock baseline architecture choice before full training.

---

Plan complete and saved to `docs/plans/2026-02-25-abnormal-crowd-behaviour-detection-implementation.md`.
