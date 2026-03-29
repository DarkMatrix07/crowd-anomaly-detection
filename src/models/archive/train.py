from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    global_batch_size: int
    per_gpu_batch_size: int
    num_workers: int
    grad_accum_steps: int
    pin_memory: bool
    use_amp: bool
    distributed_backend: str


def load_train_config(config_path: Path) -> TrainConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    train = raw.get("train", {})
    return TrainConfig(
        epochs=int(train.get("epochs", 10)),
        lr=float(train.get("lr", 1e-3)),
        global_batch_size=int(train.get("global_batch_size", 32)),
        per_gpu_batch_size=int(train.get("per_gpu_batch_size", 4)),
        num_workers=int(train.get("num_workers", 8)),
        grad_accum_steps=max(1, int(train.get("grad_accum_steps", 1))),
        pin_memory=bool(train.get("pin_memory", True)),
        use_amp=bool(train.get("use_amp", True)),
        distributed_backend=str(train.get("distributed_backend", "nccl")),
    )


def build_torchrun_command(config_path: Path, nproc_per_node: int) -> list[str]:
    return [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "scripts/train_baseline.py",
        "--config",
        str(config_path),
    ]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batch_count = 0

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if amp_enabled else None

    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(inputs)
            loss = criterion(logits, targets)
            scaled_loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(dataloader))
        if should_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        batch_count += 1

    mean_loss = total_loss / max(batch_count, 1)
    return {"loss": mean_loss}
