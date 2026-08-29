"""Evaluate a trained two-tower model against the full movie catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import mask_seen_items, single_target_metrics
from models import TwoTower


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> TwoTower:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TwoTower(
        checkpoint["num_users"],
        checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        temperature=checkpoint["temperature"],
        normalize_embeddings=checkpoint.get("normalize_embeddings", True),
        similarity=checkpoint.get("similarity", "dot"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user_id): set(group["movie_idx"].astype(int))
        for user_id, group in train.groupby("user_idx")
    }


@torch.no_grad()
def retrieve_topk(
    model: TwoTower,
    interactions: pd.DataFrame,
    seen_items: dict[int, set[int]],
    max_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve full-catalog Top-K items for every evaluation user."""
    all_item_ids = torch.arange(
        model.item_tower.embedding.num_embeddings, device=device
    )
    item_embeddings = model.item_tower(all_item_ids)
    retrieved_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for start in range(0, len(interactions), batch_size):
        batch = interactions.iloc[start : start + batch_size]
        user_ids = torch.tensor(batch["user_idx"].to_numpy(), device=device)
        target_items = torch.tensor(batch["movie_idx"].to_numpy(), device=device)

        user_embeddings = model.user_tower(user_ids)
        scores = model.score_embeddings(user_embeddings, item_embeddings)
        scores = mask_seen_items(scores, user_ids, seen_items)
        topk_items = scores.topk(max_k, dim=1).indices

        retrieved_batches.append(topk_items.cpu())
        target_batches.append(target_items.cpu())

    return torch.cat(retrieved_batches), torch.cat(target_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/two_tower.pt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    model = load_model(args.checkpoint, device)
    train = pd.read_csv(args.processed_dir / "train.csv")
    evaluation_data = pd.read_csv(args.processed_dir / f"{args.split}.csv")
    max_k = max(args.ks)
    if max_k > model.item_tower.embedding.num_embeddings:
        raise ValueError("requested K is larger than the movie catalog")

    topk_items, target_items = retrieve_topk(
        model,
        evaluation_data,
        build_seen_items(train),
        max_k,
        args.batch_size,
        device,
    )
    metrics = single_target_metrics(topk_items, target_items, args.ks)

    print(
        f"device: {device} | split: {args.split} | users: {len(evaluation_data):,} "
        f"| catalog: {model.item_tower.embedding.num_embeddings:,}"
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
