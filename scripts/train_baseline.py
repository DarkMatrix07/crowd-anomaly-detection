from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import Conv3DBaseline
from src.models.losses import build_loss
from src.models.train import load_train_config, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline anomaly model")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config")
    return parser.parse_args()


def _dummy_loader(batch_size: int) -> DataLoader:
    # Placeholder data source while dataset integration is built in later tasks.
    x = torch.randn(16, 3, 8, 24, 24)
    y = torch.randint(0, 2, (16, 1), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def main() -> int:
    args = parse_args()
    config = load_train_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv3DBaseline(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = build_loss()
    loader = _dummy_loader(batch_size=config.per_gpu_batch_size)

    for epoch in range(config.epochs):
        metrics = train_one_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=config.use_amp,
            grad_accum_steps=config.grad_accum_steps,
        )
        print(f"epoch={epoch + 1} loss={metrics['loss']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
