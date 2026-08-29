"""Compare ID-only and movie-content towers on completely unseen movies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluate import single_target_metrics
from features.prepare_movie_features import GENRES
from models import TwoTower
from models.feature_two_tower import FeatureTwoTower
from training.train import select_device, train_one_epoch


def evaluate(model, targets: pd.DataFrame, seen: dict[int, set[int]], num_items: int, device: torch.device):
    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(num_items, device=device)
        items = model.item_tower(item_ids)
        users = model.user_tower(torch.tensor(targets.user_idx.to_numpy(), device=device))
        scores = model.score_embeddings(users, items)
        for row, user in enumerate(targets.user_idx):
            if int(user) in seen:
                scores[row, list(seen[int(user)])] = -torch.inf
        topk = scores.topk(100, dim=1).indices.cpu()
    return single_target_metrics(topk, torch.tensor(targets.movie_idx.to_numpy()), [10, 50, 100])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cold-items", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cold_start_experiment.csv"))
    args = parser.parse_args()
    device = select_device()
    train = pd.read_csv(args.processed_dir / "train.csv")
    val = pd.read_csv(args.processed_dir / "val.csv")
    test = pd.read_csv(args.processed_dir / "test.csv")
    features = pd.read_csv(args.processed_dir / "movie_features.csv").sort_values("movie_idx")
    counts = train.movie_idx.value_counts()
    candidates = sorted(set(test.movie_idx) & set(counts[counts >= 5].index))
    generator = torch.Generator().manual_seed(42)
    chosen = torch.tensor(candidates)[torch.randperm(len(candidates), generator=generator)[: args.cold_items]].tolist()
    all_data = pd.concat((train, val, test), ignore_index=True)
    cold_targets = all_data[all_data.movie_idx.isin(chosen)].sort_values("timestamp").groupby("user_idx").tail(1)
    warm_train = train[~train.movie_idx.isin(chosen)]
    seen = {int(u): set(g.movie_idx.astype(int)) for u, g in warm_train.groupby("user_idx")}
    loader = DataLoader(TensorDataset(torch.tensor(warm_train.user_idx.to_numpy()), torch.tensor(warm_train.movie_idx.to_numpy())), 256, shuffle=True)
    matrix = torch.tensor(features[["release_year_scaled", *GENRES]].to_numpy(), dtype=torch.float32)
    num_users = int(all_data.user_idx.max()) + 1
    num_items = len(features)
    rows = []
    for name, model in (("id_only", TwoTower(num_users, num_items)), ("content", FeatureTwoTower(num_users, matrix))):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(args.epochs):
            loss = train_one_epoch(model, loader, optimizer, device)
        rows.append({"model": name, "cold_items": len(chosen), "targets": len(cold_targets), "final_loss": loss, **evaluate(model, cold_targets, seen, num_items, device)})
        print(rows[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
