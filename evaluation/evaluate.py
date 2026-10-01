"""Evaluate a trained two-tower model against the full movie catalog."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

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
    """Collect the movies each user interacted with during training."""
    return {
        int(user_id): set(group["movie_idx"].astype(int))
        for user_id, group in train.groupby("user_idx")
    }


def mask_seen_items(
    scores: torch.Tensor,
    user_ids: torch.Tensor,
    seen_items: dict[int, set[int]],
) -> torch.Tensor:
    """Set scores for training-seen items to negative infinity."""
    masked_scores = scores.clone()
    for row, user_id in enumerate(user_ids.tolist()):
        seen = seen_items.get(user_id, set())
        if seen:
            indices = torch.tensor(sorted(seen), device=scores.device)
            masked_scores[row, indices] = -torch.inf
    return masked_scores


def retrieve_baseline_topk(
    train: pd.DataFrame,
    interactions: pd.DataFrame,
    seen_items: dict[int, set[int]],
    num_items: int,
    max_k: int,
    batch_size: int,
    strategy: str,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve with a non-learned random or popularity ranking rule."""
    if strategy not in {"random", "popularity"}:
        raise ValueError("strategy must be 'random' or 'popularity'")
    if max_k <= 0 or max_k > num_items:
        raise ValueError("max_k must be between 1 and num_items")

    popularity = torch.bincount(
        torch.as_tensor(train["movie_idx"].to_numpy(), dtype=torch.long),
        minlength=num_items,
    ).float()
    generator = torch.Generator().manual_seed(seed)
    retrieved_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for start in range(0, len(interactions), batch_size):
        batch = interactions.iloc[start : start + batch_size]
        user_ids = torch.as_tensor(batch["user_idx"].to_numpy(), dtype=torch.long)
        target_items = torch.as_tensor(
            batch["movie_idx"].to_numpy(), dtype=torch.long
        )
        if strategy == "popularity":
            scores = popularity.unsqueeze(0).expand(len(batch), -1)
        else:
            scores = torch.rand(len(batch), num_items, generator=generator)
        scores = mask_seen_items(scores, user_ids, seen_items)
        retrieved_batches.append(scores.topk(max_k, dim=1).indices)
        target_batches.append(target_items)

    return torch.cat(retrieved_batches), torch.cat(target_batches)


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


def single_target_metrics(
    topk_items: torch.Tensor,
    target_items: torch.Tensor,
    ks: Iterable[int],
) -> dict[str, float]:
    """Calculate retrieval and rank-sensitive metrics for one target per user."""
    if topk_items.ndim != 2:
        raise ValueError("topk_items must have shape [num_users, max_k]")
    if target_items.ndim != 1 or len(target_items) != len(topk_items):
        raise ValueError("target_items must have shape [num_users]")

    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        if k <= 0 or k > topk_items.shape[1]:
            raise ValueError(f"k={k} is outside the available Top-K results")
        matches = topk_items[:, :k] == target_items[:, None]
        hits = matches.any(dim=1)
        hit_rate = hits.float().mean().item()
        rank_positions = torch.arange(
            1, k + 1, dtype=torch.float32, device=topk_items.device
        )
        reciprocal_ranks = (matches.float() / rank_positions).sum(dim=1)
        discounted_gains = (
            matches.float() / torch.log2(rank_positions + 1)
        ).sum(dim=1)
        # With one relevant item per user, recall and hit rate are identical.
        metrics[f"Recall@{k}"] = hit_rate
        metrics[f"HitRate@{k}"] = hit_rate
        metrics[f"MRR@{k}"] = reciprocal_ranks.mean().item()
        metrics[f"NDCG@{k}"] = discounted_gains.mean().item()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/two_tower_1m.pt")
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed-1m")
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-baselines", action="store_true")
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
    results = {"two_tower": single_target_metrics(topk_items, target_items, args.ks)}
    if not args.no_baselines:
        seen_items = build_seen_items(train)
        for strategy in ("random", "popularity"):
            baseline_items, baseline_targets = retrieve_baseline_topk(
                train,
                evaluation_data,
                seen_items,
                model.item_tower.embedding.num_embeddings,
                max_k,
                args.batch_size,
                strategy,
                args.seed,
            )
            results[strategy] = single_target_metrics(
                baseline_items, baseline_targets, args.ks
            )

    print(
        f"device: {device} | split: {args.split} | users: {len(evaluation_data):,} "
        f"| catalog: {model.item_tower.embedding.num_embeddings:,}"
    )
    for method, metrics in results.items():
        print(f"\n{method}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
