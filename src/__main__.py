import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Abnormal crowd behaviour detection project CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    test_cmd = subparsers.add_parser("test", help="Run the test suite")
    test_cmd.add_argument("pytest_args", nargs="*", help="Extra pytest args")

    train_cmd = subparsers.add_parser("train", help="Run model training pipeline")
    train_cmd.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training config file",
    )

    infer_cmd = subparsers.add_parser("infer", help="Run inference pipeline")
    infer_cmd.add_argument(
        "--source",
        required=False,
        default="",
        help="RTSP URL or local video path",
    )
    infer_cmd.add_argument(
        "--config",
        default="configs/infer.yaml",
        help="Inference config file",
    )

    return parser


def _run_tests(pytest_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    return subprocess.run(cmd, check=False).returncode


def _path_exists(path_value: str) -> bool:
    return Path(path_value).exists()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "test":
        return _run_tests(args.pytest_args)

    if args.command == "train":
        print(f"Training scaffold ready. Use config: {args.config}")
        if not _path_exists(args.config):
            print(f"Warning: config file not found: {args.config}")
        return 0

    if args.command == "infer":
        print(f"Inference scaffold ready. Source: {args.source or '[not set]'}")
        print(f"Using config: {args.config}")
        if not _path_exists(args.config):
            print(f"Warning: config file not found: {args.config}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
