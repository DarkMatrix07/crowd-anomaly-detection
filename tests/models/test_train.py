from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import Conv3DBaseline
from src.models.losses import build_loss
from src.models.train import load_train_config, train_one_epoch


def test_load_train_config_includes_dgx_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train:",
                "  epochs: 2",
                "  lr: 0.001",
                "  global_batch_size: 16",
                "  per_gpu_batch_size: 4",
                "  num_workers: 8",
                "  grad_accum_steps: 1",
                "  pin_memory: true",
                "  use_amp: true",
                "  distributed_backend: nccl",
            ]
        ),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.global_batch_size == 16
    assert config.per_gpu_batch_size == 4
    assert config.num_workers == 8
    assert config.grad_accum_steps == 1
    assert config.pin_memory is True
    assert config.use_amp is True
    assert config.distributed_backend == "nccl"


def test_train_one_epoch_returns_loss_metric() -> None:
    torch.manual_seed(5)
    x = torch.randn(6, 3, 8, 24, 24)
    y = torch.randint(0, 2, (6, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = Conv3DBaseline(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss()

    metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=loss_fn,
        device=torch.device("cpu"),
        use_amp=False,
        grad_accum_steps=1,
    )

    assert "loss" in metrics
    assert metrics["loss"] >= 0.0
