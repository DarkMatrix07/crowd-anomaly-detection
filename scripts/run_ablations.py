from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ablation import run_ablation_suite, select_best_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiment summary")
    parser.add_argument("--out-csv", default="docs/reports/ablation-results.csv", help="CSV output path")
    parser.add_argument("--out-md", default="docs/reports/ablation-study.md", help="Markdown output path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--window-minutes", type=float, default=4.0, help="Evaluation window duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _build_synthetic_scores(seed: int = 42) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y_true = np.array([0] * 80 + [1] * 80 + [0] * 80 + [1] * 80, dtype=np.int32)
    base = np.clip(0.25 + 0.55 * y_true + rng.normal(0.0, 0.08, size=y_true.shape[0]), 0.0, 1.0)

    variants = {
        "baseline": base,
        "no_flow": np.clip(base - 0.08 * y_true + rng.normal(0.0, 0.02, size=base.shape[0]), 0.0, 1.0),
        "no_density": np.clip(base - 0.06 * y_true + rng.normal(0.0, 0.025, size=base.shape[0]), 0.0, 1.0),
        "no_smoothing": np.clip(base + rng.normal(0.03, 0.08, size=base.shape[0]), 0.0, 1.0),
        "alternate_backbone": np.clip(base + 0.02 * y_true + rng.normal(0.0, 0.06, size=base.shape[0]), 0.0, 1.0),
    }
    return y_true, variants


def _to_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *body])


def main() -> int:
    args = parse_args()
    y_true, variants = _build_synthetic_scores(seed=args.seed)
    results = run_ablation_suite(
        y_true=y_true,
        variant_scores=variants,
        threshold=args.threshold,
        window_minutes=args.window_minutes,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)

    best = select_best_tradeoff(results, false_alert_weight=0.05)
    rounded = results.copy()
    for col in rounded.columns:
        if col != "variant":
            rounded[col] = rounded[col].astype(float).round(4)

    rows = rounded.to_dict(orient="records")
    columns = [
        "variant",
        "roc_auc",
        "pr_auc",
        "f1",
        "event_f1",
        "false_alert_rate_per_10m",
    ]
    table = _to_markdown_table(rows, columns)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join(
            [
                "# Ablation Study",
                "",
                "## Scope",
                "",
                "Synthetic placeholder run for pipeline verification. Replace with real ShanghaiTech scores after dataset training.",
                "",
                "## Results",
                "",
                table,
                "",
                "## Recommended Variant",
                "",
                f"- Best tradeoff variant: `{best['variant']}`",
                f"- Tradeoff score: `{best['tradeoff_score']:.4f}`",
                "",
                "## Next Action",
                "",
                "Re-run this script with real model outputs after dataset training and supervisor-reviewed thresholds.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"ablation_csv={out_csv}")
    print(f"ablation_md={out_md}")
    print(f"best_variant={best['variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
