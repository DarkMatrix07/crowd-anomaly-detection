import subprocess
import sys

import cv2
import numpy as np
from scipy.io import savemat


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


def _make_shanghaitech_like_sample(root, part, subset, idx, count):
    image_dir = root / part / subset / "images"
    gt_dir = root / part / subset / "ground-truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"IMG_{idx}.jpg"
    img = np.full((32, 32, 3), min(255, count), dtype=np.uint8)
    cv2.imwrite(str(image_path), img)

    gt_path = gt_dir / f"GT_IMG_{idx}.mat"
    location = np.zeros((count, 2), dtype=np.float32)
    number = np.array([[count]], dtype=np.uint16)
    info = np.empty((1, 1), dtype=[("location", "O"), ("number", "O")])
    info[0, 0] = (location, number)
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = info
    savemat(gt_path, {"image_info": image_info})


def test_train_shanghaitech_script_runs(tmp_path) -> None:
    dataset_root = tmp_path / "ShanghaiTech"
    for part in ["part_A", "part_B"]:
        for subset in ["train_data", "test_data"]:
            for idx in range(1, 6):
                _make_shanghaitech_like_sample(
                    dataset_root,
                    part=part,
                    subset=subset,
                    idx=idx,
                    count=idx * 10 + (50 if part == "part_B" else 0),
                )

    metrics_out = tmp_path / "metrics.json"
    manifest_out = tmp_path / "manifest.csv"
    model_out = tmp_path / "model.joblib"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_shanghaitech_70_30.py",
            "--dataset-root",
            str(dataset_root),
            "--epochs",
            "1",
            "--n-estimators",
            "50",
            "--metrics-out",
            str(metrics_out),
            "--manifest-out",
            str(manifest_out),
            "--model-out",
            str(model_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert metrics_out.exists()
    assert manifest_out.exists()
    assert model_out.exists()


def test_train_shanghaitech_cnn_script_runs(tmp_path) -> None:
    dataset_root = tmp_path / "ShanghaiTech"
    for part in ["part_A", "part_B"]:
        for subset in ["train_data", "test_data"]:
            for idx in range(1, 6):
                _make_shanghaitech_like_sample(
                    dataset_root,
                    part=part,
                    subset=subset,
                    idx=idx,
                    count=idx * 10 + (50 if part == "part_B" else 0),
                )

    metrics_out = tmp_path / "cnn_metrics.json"
    manifest_out = tmp_path / "cnn_manifest.csv"
    model_out = tmp_path / "cnn_model.pt"
    preds_out = tmp_path / "cnn_predictions.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_shanghaitech_cnn_70_30.py",
            "--dataset-root",
            str(dataset_root),
            "--train-ratio",
            "0.7",
            "--model-name",
            "tiny",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
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
            "scripts/train_anomaly_classifier_70_30.py",
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
