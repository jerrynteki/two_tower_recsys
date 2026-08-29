"""Export model embeddings without loading FAISS into the PyTorch process."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate import load_model, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-idx", type=int)
    args = parser.parse_args()
    device = select_device()
    model = load_model(args.checkpoint, device)
    with torch.no_grad():
        if args.user_idx is None:
            ids = torch.arange(model.item_tower.embedding.num_embeddings, device=device)
            vectors = model.item_tower(ids)
        else:
            vectors = model.user_tower(torch.tensor([args.user_idx], device=device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors.cpu().numpy().astype("float32"))


if __name__ == "__main__":
    main()
