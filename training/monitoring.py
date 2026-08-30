"""Shared TensorBoard logging helpers for training and experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


def timestamped_run_dir(base_dir: Path, name: str) -> Path:
    """Create a unique, readable directory name for one training run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / f"{timestamp}_{name}"


def log_configuration(writer: SummaryWriter, config: dict[str, Any]) -> None:
    """Show the run configuration under TensorBoard's Text tab."""
    rows = ["| Parameter | Value |", "|---|---|"]
    rows.extend(f"| {key} | {value} |" for key, value in sorted(config.items()))
    writer.add_text("run/configuration", "\n".join(rows), 0)


@torch.no_grad()
def log_model_statistics(
    writer: SummaryWriter,
    model: torch.nn.Module,
    step: int,
) -> None:
    """Track parameter and embedding magnitudes without logging huge tensors."""
    parameter_norm = torch.linalg.vector_norm(
        torch.stack([parameter.detach().float().norm() for parameter in model.parameters()])
    ).item()
    writer.add_scalar("model/parameter_norm", parameter_norm, step)

    for tower_name in ("user_tower", "item_tower"):
        tower = getattr(model, tower_name, None)
        embedding = getattr(tower, "embedding", None)
        if embedding is not None:
            row_norms = embedding.weight.detach().float().norm(dim=1)
            writer.add_scalar(
                f"model/{tower_name}_embedding_norm_mean",
                row_norms.mean().item(),
                step,
            )


def log_validation_metrics(
    writer: SummaryWriter,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Group validation metrics into readable TensorBoard series."""
    for name, value in metrics.items():
        family, k = name.split("@")
        tag = f"validation/{family.lower()}_at_{k}"
        writer.add_scalar(tag, value, step)
