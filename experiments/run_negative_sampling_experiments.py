"""Compare in-batch, uniform, popularity, and hard negative sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import InteractionDataset
from evaluation.evaluate import build_seen_items, retrieve_topk
from evaluation.metrics import single_target_metrics
from models import TwoTower
from training.negative_sampling import CatalogNegativeSampler, train_sampled_epoch
from training.train import load_catalog_sizes, select_device, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/negative_sampling_experiments.csv"))
    args = parser.parse_args()
    device = select_device()
    train = pd.read_csv(args.processed_dir / "train.csv")
    val = pd.read_csv(args.processed_dir / "val.csv")
    dataset = InteractionDataset(args.processed_dir / "train.csv")
    num_users, num_items = load_catalog_sizes(args.processed_dir)
    seen = build_seen_items(train)
    popularity = torch.bincount(torch.tensor(train.movie_idx), minlength=num_items).float()
    rows = []
    for method in ("in_batch", "uniform", "popularity", "hard"):
        torch.manual_seed(42)
        loader = DataLoader(dataset, args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
        model = TwoTower(num_users, num_items).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sampler = CatalogNegativeSampler(num_items, seen, popularity if method == "popularity" else None)
        for _ in range(args.epochs):
            if method == "in_batch":
                loss = train_one_epoch(model, loader, optimizer, device)
            else:
                pool = args.negative_count * 4 if method == "hard" else None
                loss = train_sampled_epoch(model, loader, optimizer, device, sampler, args.negative_count, pool)
        topk, targets = retrieve_topk(model, val, seen, 100, args.batch_size, device)
        metrics = single_target_metrics(topk, targets, [10, 50, 100])
        row = {"method": method, "final_loss": loss, **metrics}
        rows.append(row)
        print(row)
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
