import subprocess
import sys


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
