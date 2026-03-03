# About Project Files

## Abnormal Crowd Behaviour Detection System

A detailed explanation of every file and folder in this project — what it is, why it exists, and what it does.

---

## Directory Structure at a Glance

```
project/
├── src/                    # Core application source code
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature extraction (optical flow, crowd metrics)
│   ├── models/             # ML model definitions, training, evaluation
│   ├── risk/               # Risk scoring and alert logic
│   ├── inference/          # Real-time inference pipeline
│   └── api/                # FastAPI REST server
├── dashboard/              # Streamlit operator dashboard
├── scripts/                # Standalone runnable scripts
├── configs/                # YAML configuration files
├── tests/                  # Automated test suite
├── docs/                   # Reports and documentation
├── artifacts/              # Trained models and output files
├── data/                   # Raw and processed dataset
├── documentation/          # This documentation folder
├── START_DEMO.sh           # Demo launcher (Git Bash)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and tooling config
└── .env.example            # Environment variable template
```

---

## Root-Level Files

### `START_DEMO.sh`
**What it is:** A bash shell script that launches the interactive demo.
**Why it exists:** The `.sh` file runs in Git Bash, which has proper error display, coloured output, and always waits for the user to press Enter before closing — far more reliable than Windows CMD for running Python services.
**What it does:**
1. Changes directory to the project folder
2. Checks that Python is available in PATH
3. Checks that all required Python packages are installed (imports streamlit, cv2, joblib, sklearn)
4. If packages are missing, runs `pip install -r requirements.txt` automatically
5. Checks that the trained model file exists
6. Launches `python -m streamlit run scripts/crowd_anomaly_demo.py`
7. If anything fails, shows the error in red and waits for Enter before closing

**How to run:** Open Git Bash in the project folder and type `bash START_DEMO.sh`

---

### `requirements.txt`
**What it is:** A flat list of all Python packages the project depends on.
**Why it exists:** Allows anyone to install all dependencies with a single command: `pip install -r requirements.txt`. This ensures that everyone running the project has the exact same packages.
**Key packages and why each is needed:**
- `numpy` — array operations, numerical computing for feature vectors
- `pandas` — dataframe operations for metrics, CSV output, alert logs
- `opencv-python-headless` — video reading, frame decoding, optical flow computation, image processing
- `scikit-learn` — RandomForest classifier, train/test split, metrics (ROC-AUC, F1)
- `joblib` — saving and loading the trained model to/from a .joblib file
- `PyYAML` — reading the YAML configuration files in `configs/`
- `fastapi` — building the REST API server with automatic OpenAPI documentation
- `uvicorn` — ASGI server that runs the FastAPI application
- `httpx` — HTTP client used by the Streamlit dashboard to call the API
- `streamlit` — building the interactive web dashboard and demo UI
- `torch` — PyTorch, used by the legacy 3D CNN baseline model in `src/models/baseline.py`
- `pytest` — running the automated test suite

---

### `pyproject.toml`
**What it is:** The modern Python project configuration file (PEP 517/518).
**Why it exists:** Defines project metadata (name, version, Python version requirement), development dependencies, and tool configuration in one place. It replaces the older `setup.py` approach.
**What it contains:**
- Project name: `abnormal-crowd-detection`, version: `0.1.0`
- Python version requirement: `>=3.11`
- All dependency specifications (same as requirements.txt but in TOML format)
- CLI entry point: `crowdguard = src.__main__:main` (run the project as `python -m src`)
- Pytest configuration: test paths, filter warnings
- Build system: `flit_core`

---

### `.env.example`
**What it is:** A template file showing what environment variables the project uses.
**Why it exists:** Sensitive or environment-specific values (paths, API keys, log levels) should not be hardcoded in source code. This file shows what variables need to be set. Users copy it to `.env` and fill in their values.
**Variables:**
- `DATA_ROOT=./data/raw` — where the dataset is stored
- `DATASET_NAME=shanghaitech` — which dataset to use
- `DATASET_SUBSET=part_a` — which subset of the dataset
- `LOG_LEVEL=INFO` — logging verbosity (DEBUG/INFO/WARNING/ERROR)
- `OUTPUT_DIR=./artifacts` — where to save model files and reports

---

### `README.md`
**What it is:** The top-level project readme visible on GitHub.
**Why it exists:** Provides a quick start guide for developers. Contains installation instructions, CLI usage, how to start the API server and dashboard, and evaluation commands.

---

## `src/` — Core Source Code

This is the main application code, organised into six sub-packages. Each sub-package handles one specific concern (data, features, models, risk, inference, API), following the principle of high cohesion and low coupling.

---

### `src/__init__.py`
**What it is:** Makes `src` a Python package.
**Why it exists:** Allows other code to import from `src` using `from src.features import ...` syntax. Without this file, Python would not recognise `src` as a package.

---

### `src/__main__.py`
**What it is:** The CLI entry point for the project.
**Why it exists:** Enables running the project as a module with `python -m src`. Provides a command-line interface with subcommands for training, inference, and testing, so the system can be operated without writing Python code directly.

---

## `src/data/` — Data Loading and Preprocessing

This sub-package handles everything related to reading the dataset from disk, extracting video frames, and preparing them for the feature extraction pipeline.

### `src/data/__init__.py`
**What it is:** Package marker. Exports the main data functions for convenience importing.

---

### `src/data/ingest.py`
**What it is:** Dataset discovery and metadata building.
**Why it exists:** The ShanghaiTech dataset is a complex directory structure with multiple subsets, video files, and annotation files. This module automates the process of scanning the directory tree, discovering all videos, probing their FPS and frame count using OpenCV, detecting whether annotation files exist, and building a metadata CSV that the rest of the pipeline can consume.
**Key components:**
- `DataConfig` dataclass — holds dataset configuration (root path, name, subset, metadata CSV path). A dataclass was used for clean, typed configuration that is easy to serialise.
- `load_data_config()` — reads `configs/data.yaml` and returns a `DataConfig`. Centralises configuration loading.
- `build_metadata()` — recursively walks the dataset directory, finds all video files, and for each one: calls `_probe_video()` to get FPS/frame count, calls `_has_label()` to check for annotation files, and records all this in a list of dicts that becomes a CSV.
- `_probe_video()` — opens a video file with `cv2.VideoCapture`, reads the FPS and frame count properties, and closes it. Handles corrupt files gracefully.
- `_has_label()` — checks if any annotation file (.txt, .json, .mat, .csv, .xml) exists alongside the video. This is how the system automatically determines which videos are annotated.
- `write_metadata()` — saves the metadata to a CSV file for caching, so the directory scan doesn't need to repeat on every run.

---

### `src/data/preprocess.py`
**What it is:** Video preprocessing pipeline (resize, normalise, augment).
**Why it exists:** Raw video frames are not directly suitable for the model. They need to be resized to a consistent resolution, resampled to a consistent frame rate, normalised to float [0, 1] range, and optionally augmented to increase training data diversity.
**Key components:**
- `PreprocessConfig` dataclass — holds all preprocessing parameters: target FPS, output frame size, clip length, augmentation settings. Centralises all magic numbers.
- `_read_video()` — opens a video file with `cv2.VideoCapture` and reads all frames into a (T, H, W, C) numpy array. Handles file-not-found and empty video gracefully.
- `_resample_frames()` — downsamples frames by a stride to match the target FPS. Example: if video is 30 fps and target is 10 fps, takes every 3rd frame.
- `_apply_augmentations()` — applies random augmentations during training to increase diversity: random speed change (skip or duplicate frames), random horizontal flip, random brightness/contrast jitter. These help the model generalise to different lighting conditions and camera angles.
- `preprocess_video()` — the main pipeline: read → resample → augment → resize → convert to RGB → normalise to float [0, 1]. Returns (T, H, W, C) float32 array.

---

### `src/data/clips.py`
**What it is:** Temporal windowing utility.
**Why it exists:** The model operates on fixed-length sequences of frames (clips/windows), not raw videos of arbitrary length. This module slices a (T, H, W, C) video array into overlapping windows of a fixed length with a configurable stride.
**Key function:**
- `generate_clip_windows()` — given a (T, H, W, C) array, clip_length=30, and clip_stride=10, returns a (N, 30, H, W, C) array of N windows. The stride controls overlap: stride=10 means a new window starts every 10 frames, giving 20 frames of overlap with the previous window. This is important for ensuring no anomaly is missed between windows.

---

## `src/features/` — Feature Extraction

This sub-package extracts meaningful numerical features from raw video frames. These features are what the machine learning model actually sees — not raw pixels.

### `src/features/__init__.py`
**What it is:** Package marker.

---

### `src/features/optical_flow.py`
**What it is:** Optical flow computation module.
**Why it exists:** Optical flow (the measurement of pixel motion between frames) is the primary input signal for crowd motion analysis. This module provides a clean API for computing optical flow, with support for multiple algorithms.
**Key components:**
- `compute_optical_flow()` — the main entry point. Accepts two consecutive frames and returns a (H, W, 2) flow array. Supports "raft" algorithm (higher quality, requires torchvision) with automatic fallback to "farneback", and "farneback" directly.
- `_farneback_flow()` — computes Farneback dense optical flow using `cv2.calcOpticalFlowFarneback`. Parameters chosen for a good balance of accuracy and speed: 15×15 smoothing window, 3 pyramid levels, 3 iterations, 5th-order polynomial approximation. Dense means every pixel gets a flow vector, unlike sparse methods (Lucas-Kanade) that only track feature points.
- `save_flow_visualization()` — converts a flow field to an HSV colour image (hue = direction, value = magnitude) and saves it as a PNG. Useful for generating flow visualisations for presentations and reports.

---

### `src/features/crowd_metrics.py`
**What it is:** Motion statistics and crowd density feature extraction.
**Why it exists:** Raw optical flow fields are (H, W, 2) arrays — too high-dimensional for a Random Forest. This module distils them into a small set of interpretable scalar statistics that capture the key properties of crowd motion.
**Key components:**
- `compute_motion_statistics()` — the main function. Takes a flow field and returns a dict with:
  - `mean_magnitude`: average speed of all pixels
  - `variance_magnitude`: how uneven motion is across the frame
  - `directional_entropy`: disorder of motion directions (key anomaly signal)
  - `divergence_proxy`: crowd expansion/contraction signal
- `_directional_entropy()` — computes flow angles for every pixel, builds a 16-bin histogram of directions, and applies Shannon entropy formula. Higher entropy = more chaotic motion.
- `_divergence_proxy()` — computes the divergence of the flow field using finite differences (gradient of dx + gradient of dy). High divergence means crowds moving outward, characteristic of panic dispersal.
- `compute_density_pressure()` — estimates crowd density as the fraction of "foreground" pixels (those above an intensity threshold). A simple but effective occupancy measure.
- `compute_local_concentration_index()` — divides the frame into a grid, computes mean motion magnitude in each cell, and returns the standard deviation of these cell means. High values mean crowd motion is concentrated in some zones and absent in others (localised event).
- `build_feature_row()` — combines all above into a single feature dictionary, ready to be converted to a row in a training dataset.

---

### `src/features/anomaly_features.py`
**What it is:** The production feature extractor — the most important file in the features package.
**Why it exists:** This module defines the exact 10 features extracted per frame and the functions used to build the training dataset. Everything in the production training and inference pipeline flows through this file.
**Key components:**

- `compute_frame_features(frame, prev_frame)` — extracts 10 features from a single frame:
  1. Mean pixel intensity
  2. Std pixel intensity
  3. Laplacian variance (edge energy / sharpness)
  4. Occupancy (crowd density proxy)
  5. Mean optical flow magnitude
  6. Variance of optical flow magnitude
  7. Maximum optical flow magnitude
  8. Directional entropy
  9. Divergence proxy
  10. Temporal contrast change (mean absolute difference from previous frame)
  Returns a (10,) float32 numpy array.

- `_sorted_frame_paths(clip_dir)` — given a directory of frame images (e.g. `testing/frames/01_0130/`), returns a sorted list of Path objects in the correct temporal order. Sorting is by filename, which the ShanghaiTech dataset uses to encode frame numbers (e.g. `000.jpg`, `001.jpg`, ...).

- `build_labeled_frame_dataset(frames_root, masks_root, frame_stride, max_frames_per_clip)` — builds the training/test dataset from pre-extracted ShanghaiTech frames:
  1. Finds all clip directories
  2. For each clip, loads the corresponding ground truth mask (.npy file)
  3. Extracts frame features at the specified stride
  4. Labels each frame using the mask (0=normal, 1=anomalous)
  5. Returns (X, y) arrays where X is (N, 10) features and y is (N,) labels

- `build_normal_dataset_from_videos(training_videos_root, max_videos, frame_stride, resize)` — builds a dataset of normal-only samples from the 330 raw training videos:
  1. Finds all .avi files
  2. Opens each with cv2.VideoCapture
  3. Extracts frames at the specified stride
  4. Computes features for each frame
  5. Labels all as 0 (normal, since all training videos are normal)
  6. Returns (X, y) arrays
  This function is essential because the testing set alone does not have enough normal samples. The 330 training videos provide thousands of additional normal windows.

---

## `src/models/` — ML Model Definitions and Training

This sub-package contains the neural network architecture (for the legacy baseline), the training loop, evaluation metrics, and ablation framework.

### `src/models/__init__.py`
**What it is:** Package marker.

---

### `src/models/baseline.py`
**What it is:** A 3D convolutional neural network (CNN) architecture.
**Why it exists:** At the start of the project, a deep learning baseline was explored. This file defines a 3D CNN that processes (T, H, W) video clips directly. It was ultimately not used in production (the Random Forest outperformed it at the project scale), but it remains as a reference for future deep learning integration.
**Architecture:**
- Three blocks of Conv3d → BatchNorm3d → ReLU → MaxPool3d, doubling channels (16 → 32 → 64)
- AdaptiveAvgPool3d reduces spatial dimensions to (1, 1, 1)
- Two fully connected layers: 64 → hidden_dim → 1 with sigmoid output for anomaly probability

---

### `src/models/losses.py`
**What it is:** Custom loss functions for training neural networks.
**Why it exists:** Standard binary cross-entropy is not ideal for imbalanced datasets. Custom losses (focal loss, weighted BCE) allow up-weighting of the minority class (anomalies) during training.

---

### `src/models/train.py`
**What it is:** Training loop for neural network models.
**Why it exists:** Encapsulates the training process (forward pass, loss computation, backpropagation, optimiser step) in a clean, reusable function that supports gradient accumulation and automatic mixed precision (AMP).
**Key components:**
- `TrainConfig` dataclass — all training hyperparameters: epochs, learning rate, batch sizes, gradient accumulation steps, AMP flag, distributed backend
- `load_train_config()` — reads `configs/train.yaml`
- `train_one_epoch()` — runs one full epoch over the training dataloader with optional gradient accumulation (accumulating gradients over multiple small batches before stepping, to simulate larger batch sizes) and AMP (using float16 for faster computation on GPUs that support it)

---

### `src/models/evaluate.py`
**What it is:** Evaluation metrics computation.
**Why it exists:** A central place for all evaluation logic, ensuring metrics are computed consistently throughout the project (during training, ablation studies, and final evaluation).
**Key components:**
- `compute_binary_metrics()` — given y_true and y_scores, computes: ROC-AUC, PR-AUC, F1 at the given threshold, precision, recall, and accuracy. Uses scikit-learn's metrics functions.
- `threshold_sweep()` — evaluates performance at multiple threshold values (e.g. 0.1, 0.2, ..., 0.9) and returns a pandas DataFrame with metrics at each threshold. Used to find the optimal threshold (highest F1).
- `extract_event_segments()` — converts a binary series (0/1 per frame) into a list of (start_frame, end_frame) tuples representing contiguous anomalous segments. Used for event-level evaluation.
- `event_f1_score()` — computes F1 at the event level (did the model detect each anomalous event, regardless of exact frame timing?) using intersection-over-union matching between predicted and ground truth segments. This is a more practical metric than frame-level F1 for real deployments.

---

### `src/models/ablation.py`
**What it is:** Ablation study framework.
**Why it exists:** A clean, reusable API for running systematic experiments where one component is disabled or changed at a time, to measure its contribution to overall performance.
**Key components:**
- `run_ablation_suite()` — accepts a dict of {variant_name: score_array} and ground truth, runs `compute_binary_metrics` for each variant, and returns a DataFrame comparing all variants.
- `select_best_tradeoff()` — scores each variant on a weighted combination: `0.7 × F1 + 0.3 × event_F1 − penalty × false_alert_rate_per_10min`. This allows selecting the variant that best balances detection performance and operational reliability (low false alarm rate).

---

## `src/risk/` — Risk Scoring and Alert Logic

This sub-package takes the raw model probability and transforms it into actionable alerts.

### `src/risk/__init__.py`
**What it is:** Package marker.

---

### `src/risk/score.py`
**What it is:** Multi-signal risk score computation.
**Why it exists:** The Random Forest anomaly probability alone is not the most robust signal for triggering alerts. Combining it with direct motion measurements (flow instability, density pressure, trend acceleration) makes the system more reliable and reduces false positives from edge cases the model was not trained on.
**Key components:**
- `compute_risk_score()` — weighted combination of four signals:
  - `anomaly_score` (weight 0.40): Raw model output
  - `flow_instability` (weight 0.25): Mean variance of optical flow magnitude
  - `density_pressure` (weight 0.20): Mean occupancy across window
  - `trend_acceleration` (weight 0.15): Delta (change) in mean magnitude
  All weights configurable via `configs/risk.yaml`.
- `smooth_scores()` — applies exponential smoothing to a list of scores with configurable alpha. Used to smooth the score time series before applying thresholds.

---

### `src/risk/calibrate.py`
**What it is:** Threshold calibration from labeled data.
**Why it exists:** Instead of choosing thresholds (0.50, 0.75) arbitrarily, this module can learn them from data. Given a set of scores with ground truth labels, it finds quantile-based thresholds that separate normal from anomalous distributions.
**Key function:**
- `calibrate_thresholds()` — computes the 70th, 85th, and 95th percentile of anomalous scores as the low/medium/high thresholds. Falls back to defaults (0.30/0.60/0.85) if the anomalous class is missing.

---

### `src/risk/alerts.py`
**What it is:** Alert level assignment with hysteresis.
**Why it exists:** Converts a continuous score (0.0–1.0) into a discrete alert level (LOW/MEDIUM/HIGH) in a way that is stable and does not flicker between levels when the score is near a threshold.
**Key components:**
- `assign_alert_level()` — simple threshold comparison: score < low → "LOW", score < medium → "MEDIUM", else "HIGH".
- `apply_hysteresis()` — wraps `assign_alert_level` with hysteresis logic: once a level is reached, the score must fall below (threshold − margin) before the level drops. The margin is 0.05 by default. This prevents rapid oscillation between alert levels when the score fluctuates near a threshold boundary.

---

## `src/inference/` — Real-Time Inference Pipeline

This sub-package is the heart of the production system. It handles the real-time flow of video frames through all processing stages.

### `src/inference/__init__.py`
**What it is:** Package marker.

---

### `src/inference/stream.py`
**What it is:** Video frame iterator.
**Why it exists:** Provides a clean, unified way to iterate over video frames whether the source is a local file or an RTSP network stream. Abstracts the OpenCV boilerplate so the pipeline doesn't need to know the source type.
**Key function:**
- `iter_video_frames(source)` — a generator that opens `source` (file path or RTSP URL) with `cv2.VideoCapture` and yields (frame, timestamp) tuples until the video ends or the stream disconnects. The timestamp is computed from the frame index and video FPS.

---

### `src/inference/pipeline.py`
**What it is:** The main production inference pipeline. The most important file in the inference package.
**Why it exists:** Orchestrates the entire real-time processing chain: buffering frames, triggering window-level inference at the right cadence, fusing signals, smoothing, applying hysteresis, and emitting structured alert events. All the real-time logic is here so the rest of the system (API, dashboard) just receives clean, structured events.
**Class: `RollingInferencePipeline`**

Constructor parameters:
- `camera_id` — identifier for the camera source, included in all emitted events
- `clip_length` — number of frames per window (default 30)
- `clip_stride` — frames between window evaluations (default 10)
- `anomaly_model_fn` — callable that takes a (N, H, W, 3) frame array and returns a float probability
- `thresholds` — dict with low/medium/high values
- `smoothing_alpha` — exponential smoothing factor (default 0.35)
- `hysteresis_margin` — margin for stable alert level transitions (default 0.05)

Method `process_frame(frame)`:
1. Appends the new frame to an internal deque (maxlen = clip_length)
2. Increments frame counter
3. If `len(buffer) == clip_length` and `frame_counter % clip_stride == 0`:
   a. Extracts frame-level features from the buffer
   b. Computes optical flow and crowd metrics
   c. Calls `anomaly_model_fn(clip_array)` to get anomaly probability
   d. Computes flow_instability, density_pressure, trend_acceleration from extracted features
   e. Calls `compute_risk_score()` for the composite score
   f. Applies exponential smoothing
   g. Calls `assign_alert_level()` with hysteresis
   h. Constructs and returns an event dict: {timestamp, camera_id, risk_level, score, evidence_window}
4. Returns empty list if stride condition not met (no inference this frame)

---

### `src/inference/anomaly_model.py`
**What it is:** The wrapper that loads the trained Random Forest and makes it callable for the pipeline.
**Why it exists:** The `RollingInferencePipeline` accepts any callable as `anomaly_model_fn`. This module provides that callable by loading the trained `.joblib` model and wrapping it with the feature extraction logic, so the pipeline doesn't need to know anything about the model internals.
**Key components:**
- `load_anomaly_model_fn()` — loads `artifacts/models/shanghaitech_windowed_rf.joblib` using joblib. Critically, sets `model.n_jobs = 1` before returning. This is necessary because RandomForest with `n_jobs=-1` (multi-core) hangs indefinitely on Windows when called from within a Streamlit thread due to Python's multiprocessing limitations on Windows. Returns a closure that calls `_clip_to_window_features` followed by `model.predict_proba`.
- `_clip_to_window_features(clip)` — converts a (N, H, W, 3) uint8 clip array to the (1, 40) feature vector:
  1. Resizes each frame to 320×240 (the resolution used during training)
  2. Extracts 10 frame features for each frame using `compute_frame_features`
  3. Aggregates using mean, std, max, delta across frames
  4. Returns (1, 40) float32 array ready for `predict_proba`

---

## `src/api/` — FastAPI REST Server

This sub-package is the backend service that receives alerts from the inference pipeline and serves them to the operator dashboard.

### `src/api/__init__.py`
**What it is:** Package marker.

---

### `src/api/schemas.py`
**What it is:** Pydantic data models for API request/response validation.
**Why it exists:** FastAPI uses Pydantic models to automatically validate incoming JSON, serialise outgoing JSON, and generate OpenAPI documentation. Defining schemas here separates the data contract from the application logic.
**Key classes:**
- `AlertCreate` — the data required to create a new alert: timestamp (datetime), camera_id (str), risk_level (Literal["LOW", "MEDIUM", "HIGH"]), score (float 0–1), evidence_window (list of 2 ints). Using `Literal` ensures only valid alert levels are accepted.
- `AlertRecord` — `AlertCreate` plus: id (int), acknowledged_by (str|None), ack_note (str|None), acknowledged_at (datetime|None). Represents a full alert as stored in the database.
- `AlertAckRequest` — the data needed to acknowledge an alert: operator_name, note.
- `ThresholdProfile` — profile_name, low, medium, high threshold values.
- `TimelinePoint` — a single point in the risk score timeline: timestamp, score.
- `DashboardSummaryResponse` — the full dashboard payload: alerts list, timeline list, level_counts dict.

---

### `src/api/app.py`
**What it is:** The FastAPI application — the complete REST API server.
**Why it exists:** Provides a persistent, network-accessible service that stores alerts and serves them to the operator dashboard. Decouples the inference pipeline from the display layer — multiple dashboards on different machines can connect to the same API.
**Database:**
- Uses SQLite (`artifacts/alerts.db`) for alert persistence
- Schema: `alerts` table (id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end, acknowledged, acknowledged_by, ack_note, acknowledged_at) and `threshold_config` table
- Uses `row_factory = sqlite3.Row` for dict-like access to rows
- Auto-migrates: checks if acknowledgment columns exist and adds them if not (for upgrades from older schema)
- All queries use `?` parameterised placeholders (SQL injection protection)

**Key endpoints:**
- `GET /health` — health check, returns `{"status": "ok"}`. Used by the launcher scripts to verify the server is running.
- `GET /status` — current system status including total alert count.
- `POST /alerts` — accepts an `AlertCreate` body, inserts it into SQLite, returns the full `AlertRecord` with the auto-generated ID.
- `GET /alerts?limit=50&camera_id=cam_01` — returns paginated, optionally camera-filtered list of recent alerts in reverse chronological order.
- `POST /alerts/{alert_id}/ack` — marks an alert as acknowledged, records operator name, note, and timestamp.
- `GET /config/thresholds` — returns current threshold profile from database.
- `PUT /config/thresholds` — updates the threshold profile. The database persists this so thresholds survive API restarts.
- `GET /dashboard/summary?limit=100&camera_id=cam_01` — the main endpoint for the dashboard. Returns alert counts by level, a timeline of recent scores, and the full alert list in one request (minimises dashboard round-trips).

---

## `dashboard/` — Streamlit Operator Dashboard

### `dashboard/app.py`
**What it is:** The full operator monitoring dashboard.
**Why it exists:** Provides a user-friendly interface for security operators to monitor alert status across all cameras, adjust sensitivity thresholds, and acknowledge alerts — without needing to interact with the API directly.
**What it shows:**
- Connection status (API URL, currently connected)
- Camera filter dropdown (monitor one camera or all cameras)
- Three metric cards: count of LOW / MEDIUM / HIGH alerts in the window
- Threshold control panel with profile presets (default/strict/relaxed) and custom sliders
- Risk timeline chart (score over time, from the `/dashboard/summary` endpoint)
- Alert history table (full alert records)
- Acknowledgment form (enter alert ID, operator name, note)
**How it works:** Every time the user interacts with the UI, Streamlit re-runs the entire script from top to bottom. The script calls the API via `DashboardApiClient` to get fresh data on each run. This is Streamlit's standard reactive model.

---

### `dashboard/client.py`
**What it is:** HTTP client wrapper for the dashboard to call the API.
**Why it exists:** Keeps all API communication logic in one place, separate from the dashboard display logic. The dashboard just calls methods like `client.summary()` without knowing about HTTP, JSON parsing, or error handling.
**Key methods:**
- `_request(method, path, **kwargs)` — the internal HTTP call using `httpx.Client`. Raises on HTTP errors. Timeout is 10 seconds to prevent the dashboard from hanging.
- `health()` — `GET /health`
- `summary(limit, camera_id)` — `GET /dashboard/summary`
- `get_thresholds()` — `GET /config/thresholds`
- `update_thresholds(profile_name, low, medium, high)` — `PUT /config/thresholds`
- `acknowledge_alert(alert_id, operator_name, note)` — `POST /alerts/{id}/ack`

---

## `scripts/` — Standalone Runnable Scripts

These are the executable scripts that perform specific tasks. They are not imported by other modules — they are run directly.

### `scripts/crowd_anomaly_demo.py`
**What it is:** The interactive Streamlit visual demo application. The primary deliverable for client and supervisor presentations.
**Why it exists:** The full production stack (API + dashboard + live feed) requires three separate terminals and some technical setup. The demo provides a complete, self-contained experience that runs with a single command and requires no API server.
**How it works:**
1. **Sidebar:** Mode selection (Demo Clips / Upload Video), clip/file selector, FPS slider, GT toggle, threshold info, model info.
2. **Phase 1 (pre-scoring):** Before playback begins, the entire clip is analysed. Every 10 frames, a 30-frame window is feature-extracted and scored by the Random Forest. All scores are stored in a list indexed by frame number. A progress bar shows analysis progress.
3. **Phase 2 (smooth playback):** Frames are played back at the selected FPS. For each frame, the pre-computed score is retrieved (no inference during playback = no lag). The frame is displayed with an OpenCV overlay (score bar, level text, frame counter, GT label). The score chart and alert log update in real time.
4. **Upload mode:** User uploads a video file. The file bytes are saved to a temp file, opened with `cv2.VideoCapture`, and frames are extracted into memory (capped at 1500 frames). The same scoring and playback pipeline runs on these frames.
**Theme:** Dark professional (GitHub dark mode colours, custom CSS injected via `st.markdown`).

---

### `scripts/train_production_model.py`
**What it is:** The main model training script. This is how the production model was created.
**Why it exists:** Encapsulates the complete training pipeline so the model can be retrained with different parameters, on new data, or for ablation variants, with a single command.
**Command-line arguments:**
- `--test-ratio` — fraction of test clips used for validation (default 0.20)
- `--window-stride` — stride between training windows (default 5)
- `--frame-stride-clips` — stride for sampling frames within clips (default 5)
- `--max-train-videos` — max normal training videos to use (0 = all)
- `--n-estimators` — number of trees in the forest (default 300)
- `--classifier` — "rf" (Random Forest) or "gbt" (Gradient Boosting)
- `--no-flow` — flag to zero out optical flow features (for ablation)
- `--model-out` — output path for the .joblib model file
- `--metrics-out` — output path for the JSON metrics file
**Training process:**
1. Calls `build_labeled_frame_dataset()` to get frame features from test clips
2. Groups frames into 30-frame windows with majority-vote labels
3. Calls `build_normal_dataset_from_videos()` to add normal samples
4. Downsamples normal class to 2× anomaly count
5. Splits into train/test sets
6. Trains RandomForest or GBT with class balancing
7. Sets `n_jobs=1` for Windows compatibility
8. Finds optimal threshold via F1 sweep
9. Saves model with joblib, saves metrics as JSON, saves predictions as CSV

---

### `scripts/run_ablations.py`
**What it is:** Ablation study runner.
**Why it exists:** Automates running all four model variants (RF W=30, GBT W=30, RF no-flow, RF W=15) and comparing their performance in a single script call.
**What it does:** Calls `train_production_model.py` internally for each variant with different parameters, collects metrics, and outputs a comparison table in both CSV and Markdown format. Results are saved to `docs/reports/ablation-results.csv` and `docs/reports/ablation-study.md`.

---

### `scripts/evaluate_model.py`
**What it is:** Threshold sweep evaluation script.
**Why it exists:** After training, finding the optimal operating threshold is important. This script runs the trained model's predictions through `threshold_sweep()` across many threshold values and saves the results to a CSV, allowing analysis of the precision/recall tradeoff.

---

### `scripts/live_feed_demo.py`
**What it is:** Live feed simulation for the full production stack.
**Why it exists:** Provides a way to demonstrate the complete system (API + dashboard) by replaying a real ShanghaiTech clip through the `RollingInferencePipeline` and posting real alerts to the API, which then appear on the operator dashboard in real time.
**How to run:**
1. Start API: `uvicorn src.api.app:app --port 8000`
2. Start Dashboard: `streamlit run dashboard/app.py`
3. Run: `python scripts/live_feed_demo.py --clip-id 01_0130`
The script processes each frame through the pipeline, displays coloured alerts in the terminal (green/yellow/red), and POSTs each MEDIUM/HIGH alert to the API.

---

### `scripts/single_clip_demo.py`
**What it is:** Batch scoring of a single clip without visualisation.
**Why it exists:** Quick way to score a clip and get per-frame predictions as a CSV, useful for analysis, debugging, or generating data for reports.

---

### `scripts/train_baseline.py`
**What it is:** Training script for the 3D CNN baseline model (legacy).
**Why it exists:** An earlier exploration of deep learning approaches. Kept for reference and potential future use. Not used in the production pipeline.

---

### `scripts/train_anomaly_classifier.py`
**What it is:** An alternative training pipeline (earlier iteration).
**Why it exists:** An earlier version of the training pipeline before `train_production_model.py` was developed. Kept for reference.

---

### `scripts/prepare_data.py`
**What it is:** Data preparation utility.
**Why it exists:** Scans the dataset, builds the metadata CSV, and prepares directories for output artifacts. Should be run once when setting up the project on a new machine.

---

### `scripts/run_inference.py`
**What it is:** Batch inference on multiple clips.
**Why it exists:** Runs the full inference pipeline on multiple clips in sequence, outputting predictions to CSV. Useful for bulk evaluation or generating predictions for reports.

---

## `configs/` — Configuration Files

All system parameters are externalised into YAML files here, following the principle that hardcoded values should be avoided. Changing a threshold, window size, or model path only requires editing a YAML file — no code changes needed.

### `configs/data.yaml`
**Purpose:** Dataset loading configuration.
**Used by:** `src/data/ingest.py` via `load_data_config()`
**Key parameters:** dataset name, subset (part_a/part_b/both), data root path, metadata CSV path, preprocessing settings (target FPS, frame size, clip length, augmentation).

### `configs/train.yaml`
**Purpose:** Neural network training hyperparameters (for the baseline model).
**Used by:** `src/models/train.py` via `load_train_config()`
**Key parameters:** epochs, learning rate, batch sizes (global and per-GPU), gradient accumulation steps, AMP (mixed precision), distributed backend.

### `configs/risk.yaml`
**Purpose:** Risk scoring and alerting parameters.
**Used by:** `src/risk/score.py`, `src/risk/alerts.py`, `src/inference/pipeline.py`
**Key parameters:** signal weights (anomaly/flow/density/trend), alert thresholds (low/medium/high), smoothing alpha, hysteresis margin.

### `configs/infer.yaml`
**Purpose:** Inference pipeline parameters.
**Used by:** `src/inference/pipeline.py`, `scripts/run_inference.py`
**Key parameters:** clip length (window size), clip stride, max frames to process, camera ID.

---

## `tests/` — Automated Test Suite

### `tests/test_smoke.py`
**What it is:** Basic smoke tests that verify the package imports and CLI work.
**Why it exists:** The first and most fundamental tests — if these fail, nothing else can possibly work. A smoke test that imports the package catches broken __init__.py files, missing dependencies, and syntax errors immediately.
**Tests:**
- `test_package_import_smoke()` — imports `src` and verifies it has a `__version__` attribute
- `test_module_cli_help_smoke()` — runs `python -m src --help` and verifies it exits with code 0

### `tests/data/`, `tests/features/`, `tests/models/`, `tests/risk/`, `tests/inference/`, `tests/api/`, `tests/scripts/`
**What they are:** Module-specific test directories containing unit and integration tests for each source sub-package.
**Why they exist:** Each module has its own test directory to keep tests organised and make it easy to run tests for a specific component (`pytest tests/features/`).

---

## `docs/` — Documentation and Reports

### `docs/EXECUTIVE_SUMMARY.md`
**Audience:** Client and non-technical stakeholders.
**Purpose:** Explains the system in plain English — what it does, how well it performs, how to run the demo, and what the client receives. No technical jargon.

### `docs/reports/baseline-eval.md`
**Purpose:** Detailed technical evaluation report with real metrics from the trained model. Includes confusion matrix, scene-level AUC breakdown, feature importances, and failure analysis.

### `docs/reports/ablation-study.md`
**Purpose:** Results from the four ablation experiments. Documents which components contribute most to performance.

### `docs/reports/final-results.md`
**Purpose:** Summary of the complete project status, model performance, key training commands, and remaining tasks.

### `docs/reports/flow-visualization.md`
**Purpose:** Notes and naming conventions for optical flow visualisation figures (for supervisor presentations). Figures not yet generated.

### `docs/reports/baseline-threshold-sweep.csv`
**Purpose:** Metrics at every threshold value from 0.1 to 0.9. Used to select the optimal operating threshold (0.537).

### `docs/reports/ablation-results.csv`
**Purpose:** Machine-readable results from all ablation variants.

---

## `artifacts/` — Model and Output Artifacts

### `artifacts/models/shanghaitech_windowed_rf.joblib`
**What it is:** The production trained model — a serialised scikit-learn RandomForestClassifier.
**Why joblib:** joblib is the standard format for scikit-learn models. It uses numpy's efficient binary format for storing large arrays (the 300 decision trees), resulting in a compact 5.3 MB file that loads in under 1 second.
**How to load:**
```python
import joblib
model = joblib.load("artifacts/models/shanghaitech_windowed_rf.joblib")
proba = model.predict_proba(X)[0, 1]  # anomaly probability
```

### `artifacts/models/shanghaitech_windowed_gbt.joblib`
**What it is:** Gradient Boosted Trees variant trained for ablation comparison. ROC-AUC 0.8178. Not used in production.

### `artifacts/models/shanghaitech_ablation_noflow.joblib`
**What it is:** Random Forest trained without optical flow features (features 5–9 zeroed). ROC-AUC 0.8016. Used to measure the contribution of optical flow.

### `artifacts/models/shanghaitech_ablation_w15.joblib`
**What it is:** Random Forest trained with W=15 frame windows instead of W=30. ROC-AUC 0.7449. Used to measure the contribution of temporal window size.

### `artifacts/reports/`
**What they are:** JSON metrics files and CSV prediction files for each model variant. JSON files contain the full metrics dict (ROC-AUC, PR-AUC, F1, etc.). CSV files contain per-window predictions (frame index, y_true, y_score, y_pred).

### `artifacts/alerts.db`
**What it is:** SQLite database storing all alerts raised during system operation.
**Tables:** `alerts` (all alert records with acknowledgment info), `threshold_config` (current threshold settings).
**How to inspect:**
```bash
sqlite3 artifacts/alerts.db "SELECT * FROM alerts LIMIT 10;"
```

---

## `data/` — Dataset

### `data/raw/shanghaitech/shanghaitech/testing/frames/`
**What it is:** Pre-extracted JPEG frames from the 60 ShanghaiTech test clips. Each sub-directory (e.g. `01_0130/`) contains hundreds of numbered frames (`000.jpg`, `001.jpg`, ...).
**Why pre-extracted:** OpenCV can read individual JPEGs faster than seeking through compressed video files. Pre-extracting frames also makes the frame-stride sampling in `build_labeled_frame_dataset()` simpler and faster.

### `data/raw/shanghaitech/shanghaitech/testing/test_frame_mask/`
**What it is:** Ground truth annotation files. Each file (e.g. `01_0130.npy`) is a numpy binary array where 1 indicates an anomalous frame and 0 indicates normal.
**How used:** Loaded by `build_labeled_frame_dataset()` to assign labels to extracted windows.

### `data/raw/shanghaitech/shanghaitech/training/videos/`
**What it is:** 330 raw `.avi` video files of normal crowd behaviour from 13 campus scenes.
**How used:** Read by `build_normal_dataset_from_videos()` to extract normal training samples. No annotation files exist here — all training videos are normal by design (the ShanghaiTech training set contains only normal footage).
