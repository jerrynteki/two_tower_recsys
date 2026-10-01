"""Epoch-boundary checkpoints and reproducible random-state restoration."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_random_state(
    loader_generator: torch.Generator,
    sampler_generator: torch.Generator,
) -> dict[str, Any]:
    # Store NumPy's array as a list so torch.load(weights_only=True) can load it.
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
        "sampler": sampler_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_random_state(
    state: dict[str, Any],
    loader_generator: torch.Generator,
    sampler_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    loader_generator.set_state(state["loader"].cpu())
    sampler_generator.set_state(state["sampler"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint atomically, keeping the previous one on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
