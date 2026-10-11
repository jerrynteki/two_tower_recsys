"""Evaluate a saved LightGCN checkpoint using full-catalog ranking."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from evaluation.evaluate import build_seen_items, mask_seen_items, multi_target_metrics
from models.lightgcn import LightGCN, normalized_bipartite_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lightgcn_ml1m.pt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed-1m"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train = pd.read_csv(args.processed_dir / "train.csv")
    evaluation_data = pd.read_csv(args.processed_dir / f"{args.split}.csv")
    history = train
    if args.split == "test":
        validation = pd.read_csv(args.processed_dir / "val.csv")
        history = pd.concat((train, validation), ignore_index=True)
    adjacency = normalized_bipartite_adjacency(
        checkpoint["num_users"], checkpoint["num_items"],
        torch.tensor(train["user_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(train["movie_idx"].to_numpy(), dtype=torch.long),
    )
    model = LightGCN(
        checkpoint["num_users"], checkpoint["num_items"], adjacency,
        embedding_dim=checkpoint["embedding_dim"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    user_embeddings, item_embeddings = model.propagated_embeddings()
    seen_items = build_seen_items(history)
    relevant_items = build_seen_items(evaluation_data)
    user_ids = torch.tensor(sorted(relevant_items), dtype=torch.long)
    topk_parts = []
    for start in range(0, len(user_ids), args.batch_size):
        batch_users = user_ids[start : start + args.batch_size]
        scores = user_embeddings[batch_users] @ item_embeddings.T
        scores = mask_seen_items(scores, batch_users, seen_items)
        topk_parts.append(scores.topk(max(args.ks), dim=1).indices)
    metrics = multi_target_metrics(
        torch.cat(topk_parts), user_ids, relevant_items, args.ks
    )
    print(f"split: {args.split} | users: {len(user_ids):,} | catalog: {checkpoint['num_items']:,}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
