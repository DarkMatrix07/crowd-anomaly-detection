import subprocess
import sys

import cv2
import numpy as np


def test_train_baseline_script_runs_with_config() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_baseline.py", "--config", "configs/train.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_evaluate_model_script_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_model.py",
            "--ckpt",
            "dummy",
            "--out",
            "docs/reports/baseline-threshold-sweep.csv",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_ablations_script_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablations.py",
            "--out-csv",
            "docs/reports/ablation-results.csv",
            "--out-md",
            "docs/reports/ablation-study.md",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_anomaly_classifier_script_runs(tmp_path) -> None:
    dataset_root = tmp_path / "shanghaitech"
    frames_root = dataset_root / "testing" / "frames" / "01_0001"
    masks_root = dataset_root / "testing" / "test_frame_mask"
    frames_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    masks = []
    for idx in range(20):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        if idx >= 10:
            img[15:30, 20 + (idx % 5) : 35 + (idx % 5), :] = 255
            masks.append(1)
        else:
            masks.append(0)
        cv2.imwrite(str(frames_root / f"{idx:03d}.jpg"), img)
    np.save(masks_root / "01_0001.npy", np.array(masks, dtype=np.uint8))

    metrics_out = tmp_path / "anomaly_metrics.json"
    manifest_out = tmp_path / "anomaly_manifest.csv"
    model_out = tmp_path / "anomaly_model.joblib"
    preds_out = tmp_path / "anomaly_predictions.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_anomaly_classifier.py",
            "--dataset-root",
            str(dataset_root),
            "--train-ratio",
            "0.7",
            "--frame-stride",
            "1",
            "--max-frames-per-clip",
            "30",
            "--n-estimators",
            "50",
            "--metrics-out",
            str(metrics_out),
            "--manifest-out",
            str(manifest_out),
            "--model-out",
            str(model_out),
            "--predictions-out",
            str(preds_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert metrics_out.exists()
    assert manifest_out.exists()
    assert model_out.exists()
    assert preds_out.exists()


def test_demo_anomaly_clip_script_runs(tmp_path) -> None:
    dataset_root = tmp_path / "shanghaitech"
    frames_root = dataset_root / "testing" / "frames" / "01_0001"
    masks_root = dataset_root / "testing" / "test_frame_mask"
    frames_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    masks = []
    for idx in range(20):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        if idx >= 10:
            img[15:30, 20 + (idx % 5) : 35 + (idx % 5), :] = 255
            masks.append(1)
        else:
            masks.append(0)
        cv2.imwrite(str(frames_root / f"{idx:03d}.jpg"), img)
    np.save(masks_root / "01_0001.npy", np.array(masks, dtype=np.uint8))

    # Train a tiny quick model first for demo script input.
    model_out = tmp_path / "anomaly_model.joblib"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_anomaly_classifier.py",
            "--dataset-root",
            str(dataset_root),
            "--train-ratio",
            "0.7",
            "--frame-stride",
            "1",
            "--max-frames-per-clip",
            "30",
            "--n-estimators",
            "30",
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--manifest-out",
            str(tmp_path / "manifest.csv"),
            "--predictions-out",
            str(tmp_path / "preds.csv"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    demo_csv = tmp_path / "demo_preds.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/single_clip_demo.py",
            "--model-path",
            str(model_out),
            "--dataset-root",
            str(dataset_root),
            "--clip-id",
            "01_0001",
            "--frame-stride",
            "1",
            "--max-frames",
            "20",
            "--out-csv",
            str(demo_csv),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert demo_csv.exists()
