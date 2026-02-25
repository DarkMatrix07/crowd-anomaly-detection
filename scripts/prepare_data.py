from __future__ import annotations

import argparse
from pathlib import Path

from src.data.ingest import load_data_config, write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset artifacts")
    parser.add_argument(
        "--stage",
        choices=("ingest", "all"),
        default="ingest",
        help="Preparation stage to run",
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Data config path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_data_config(Path(args.config))
    output = write_metadata(config)
    print(f"Wrote metadata: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
