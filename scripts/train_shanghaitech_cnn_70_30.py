from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.shanghaitech_count import collect_samples, extract_count_from_mat, split_indices
from src.models.shanghaitech_cnn import (
    ShanghaiTechCountDataset,
    build_model,
    collate_count_batches,
    set_seed,
    tolerance_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShanghaiTech CNN crowd counting model with 70/30 split")
    parser.add_argument("--dataset-root", default="ShanghaiTech", help="Path to ShanghaiTech root")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-name", choices=("resnet18", "tiny"), default="resnet18")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--model-out", default="artifacts/models/shanghaitech_cnn_70_30.pt")
    parser.add_argument("--metrics-out", default="artifacts/reports/shanghaitech_cnn_70_30_metrics.json")
    parser.add_argument("--manifest-out", default="data/interim/shanghaitech_cnn_70_30_manifest.csv")
    parser.add_argument("--predictions-out", default="artifacts/reports/shanghaitech_cnn_70_30_predictions.csv")
    return parser.parse_args()


def _split_train_val(train_idx: list[int], seed: int, val_ratio: float) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.asarray(train_idx, dtype=np.int64)
    rng.shuffle(idx)
    val_count = int(round(len(idx) * val_ratio))
    val_count = max(1, min(val_count, len(idx) - 1))
    val_idx = sorted(idx[:val_count].tolist())
    train_main = sorted(idx[val_count:].tolist())
    return train_main, val_idx


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    losses: list[float] = []
    y_true_counts: list[float] = []
    y_pred_counts: list[float] = []
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for images, counts_log, ids, parts in loader:
            images = images.to(device=device, dtype=torch.float32)
            counts_log = counts_log.to(device=device, dtype=torch.float32)
            preds_log = model(images)
            loss = criterion(preds_log, counts_log)
            losses.append(float(loss.item()))

            pred_counts = torch.expm1(preds_log.squeeze(1)).cpu().numpy()
            true_counts = torch.expm1(counts_log.squeeze(1)).cpu().numpy()
            y_pred_counts.extend(pred_counts.tolist())
            y_true_counts.extend(true_counts.tolist())
            for sample_id, part, true_c, pred_c in zip(ids, parts, true_counts, pred_counts):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "part": part,
                        "true_count": float(true_c),
                        "pred_count": float(pred_c),
                        "abs_error": float(abs(pred_c - true_c)),
                    }
                )

    y_true = np.asarray(y_true_counts, dtype=np.float32)
    y_pred = np.asarray(y_pred_counts, dtype=np.float32)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "tolerance_accuracy": float(tolerance_accuracy(y_true, y_pred)),
    }
    return float(np.mean(losses) if losses else 0.0), metrics, rows


def _save_manifest(
    samples,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    out_csv: Path,
) -> None:
    split_lookup = {idx: "train" for idx in train_idx}
    split_lookup.update({idx: "val" for idx in val_idx})
    split_lookup.update({idx: "test" for idx in test_idx})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "part", "count", "split", "image_path"])
        for idx, sample in enumerate(samples):
            split = split_lookup[idx]
            writer.writerow([sample.sample_id, sample.part, extract_count_from_mat(sample.gt_path), split, sample.image_path])


def _save_predictions(rows: list[dict[str, float | str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "part", "true_count", "pred_count", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    samples = collect_samples(Path(args.dataset_root))
    parts = [s.part for s in samples]
    train_idx, test_idx = split_indices(parts, train_ratio=args.train_ratio, seed=args.seed)
    train_main_idx, val_idx = _split_train_val(train_idx, seed=args.seed, val_ratio=args.val_ratio)

    train_ds = ShanghaiTechCountDataset(samples, train_main_idx, image_size=args.image_size, augment=True)
    val_ds = ShanghaiTechCountDataset(samples, val_idx, image_size=args.image_size, augment=False)
    test_ds = ShanghaiTechCountDataset(samples, test_idx, image_size=args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_count_batches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_count_batches,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_count_batches,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name=args.model_name, pretrained=args.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, counts_log, _, _ in train_loader:
            images = images.to(device=device, dtype=torch.float32)
            counts_log = counts_log.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            preds_log = model(images)
            loss = criterion(preds_log, counts_log)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses) if train_losses else 0.0)
        val_loss, val_metrics, _ = _evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_tolerance_accuracy": val_metrics["tolerance_accuracy"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_tol_acc={val_metrics['tolerance_accuracy']:.4f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early_stop_epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_metrics, test_rows = _evaluate(model, test_loader, device, criterion)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "dataset_root": str(args.dataset_root),
        "model_name": args.model_name,
        "pretrained": bool(args.pretrained),
        "device": str(device),
        "samples_total": len(samples),
        "train_samples": len(train_main_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_ratio_overall": len(train_idx) / len(samples),
        "test_ratio_overall": len(test_idx) / len(samples),
        "metrics_test": test_metrics,
        "history": history,
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    _save_manifest(samples, train_main_idx, val_idx, test_idx, out_csv=Path(args.manifest_out))
    _save_predictions(test_rows, out_csv=Path(args.predictions_out))

    print(f"test_mae={test_metrics['mae']:.4f}")
    print(f"test_rmse={test_metrics['rmse']:.4f}")
    print(f"test_r2={test_metrics['r2']:.4f}")
    print(f"test_tolerance_accuracy={test_metrics['tolerance_accuracy']:.4f}")
    print(f"metrics={metrics_out}")
    print(f"model={model_out}")
    print(f"manifest={Path(args.manifest_out)}")
    print(f"predictions={Path(args.predictions_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
