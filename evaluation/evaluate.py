"""Evaluate a trained two-tower model against the full movie catalog."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

from models import TwoTower
from models.feature_two_tower import HybridFeatureTwoTower


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture", "mlp")
    common = {
        "embedding_dim": checkpoint["embedding_dim"],
        "temperature": checkpoint["temperature"],
        "normalize_embeddings": checkpoint.get("normalize_embeddings", True),
        "similarity": checkpoint.get("similarity", "dot"),
    }
    if architecture == "feature_hybrid":
        state_dict = checkpoint["model_state_dict"]
        model = HybridFeatureTwoTower(
            checkpoint["num_users"],
            state_dict["item_tower.feature_matrix"],
            **common,
        ).to(device)
    else:
        model = TwoTower(
            checkpoint["num_users"],
            checkpoint["num_items"],
            architecture=architecture,
            **common,
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
    user_ids = torch.as_tensor(
        interactions["user_idx"].to_numpy(), dtype=torch.long
    )
    retrieved = retrieve_baseline_topk_for_users(
        train,
        user_ids,
        seen_items,
        num_items,
        max_k,
        batch_size,
        strategy,
        seed,
    )
    targets = torch.as_tensor(
        interactions["movie_idx"].to_numpy(), dtype=torch.long
    )
    return retrieved, targets


def retrieve_baseline_topk_for_users(
    train: pd.DataFrame,
    user_ids: torch.Tensor,
    seen_items: dict[int, set[int]],
    num_items: int,
    max_k: int,
    batch_size: int,
    strategy: str,
    seed: int = 42,
) -> torch.Tensor:
    """Retrieve a baseline Top-K list once for each supplied user."""
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

    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start : start + batch_size]
        if strategy == "popularity":
            scores = popularity.unsqueeze(0).expand(len(batch_user_ids), -1)
        else:
            scores = torch.rand(
                len(batch_user_ids), num_items, generator=generator
            )
        scores = mask_seen_items(scores, batch_user_ids, seen_items)
        retrieved_batches.append(scores.topk(max_k, dim=1).indices)

    return torch.cat(retrieved_batches)


@torch.no_grad()
def retrieve_topk_for_users(
    model: TwoTower,
    user_ids: torch.Tensor,
    seen_items: dict[int, set[int]],
    max_k: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Retrieve one full-catalog Top-K list for each supplied user."""
    all_item_ids = torch.arange(model.num_items, device=device)
    item_embeddings = model.item_tower(all_item_ids)
    retrieved_batches: list[torch.Tensor] = []

    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start : start + batch_size].to(device)

        user_embeddings = model.user_tower(batch_user_ids)
        scores = model.score_embeddings(user_embeddings, item_embeddings)
        scores = mask_seen_items(scores, batch_user_ids, seen_items)
        topk_items = scores.topk(max_k, dim=1).indices

        retrieved_batches.append(topk_items.cpu())

    return torch.cat(retrieved_batches)


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


def multi_target_metrics(
    topk_items: torch.Tensor,
    user_ids: torch.Tensor,
    relevant_items: dict[int, set[int]],
    ks: Iterable[int],
) -> dict[str, float]:
    """Calculate per-user Recall and NDCG for multiple targets."""
    if topk_items.ndim != 2:
        raise ValueError("topk_items must have shape [num_users, max_k]")
    if user_ids.ndim != 1 or len(user_ids) != len(topk_items):
        raise ValueError("user_ids must have shape [num_users]")

    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        if k <= 0 or k > topk_items.shape[1]:
            raise ValueError(f"k={k} is outside the available Top-K results")
        recalls: list[float] = []
        ndcgs: list[float] = []
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2).float())

        for row, user_id in enumerate(user_ids.tolist()):
            targets = relevant_items.get(int(user_id), set())
            if not targets:
                raise ValueError(f"user {user_id} has no relevant items")
            relevance = torch.tensor(
                [int(item in targets) for item in topk_items[row, :k].tolist()],
                dtype=torch.float32,
            )
            recalls.append(float(relevance.sum().item() / len(targets)))
            dcg = float((relevance * discounts).sum().item())
            ideal_count = min(len(targets), k)
            idcg = float(discounts[:ideal_count].sum().item())
            ndcgs.append(dcg / idcg)

        metrics[f"Recall@{k}"] = sum(recalls) / len(recalls)
        metrics[f"NDCG@{k}"] = sum(ndcgs) / len(ndcgs)
    return metrics


def evaluate_retrieval_metrics(
    model: TwoTower,
    interactions: pd.DataFrame,
    seen_items: dict[int, set[int]],
    ks: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multi-positive full-catalog retrieval for each user."""
    ks = list(ks)
    relevant_items = build_seen_items(interactions)
    user_ids = torch.tensor(sorted(relevant_items), dtype=torch.long)
    topk_items = retrieve_topk_for_users(
        model,
        user_ids,
        seen_items,
        max(ks),
        batch_size,
        device,
    )
    return multi_target_metrics(topk_items, user_ids, relevant_items, ks)


def evaluate_baseline_metrics(
    train: pd.DataFrame,
    interactions: pd.DataFrame,
    seen_items: dict[int, set[int]],
    num_items: int,
    ks: Iterable[int],
    batch_size: int,
    strategy: str,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a random or popularity ranking with multi-positive metrics."""
    ks = list(ks)
    relevant_items = build_seen_items(interactions)
    user_ids = torch.tensor(sorted(relevant_items), dtype=torch.long)
    topk_items = retrieve_baseline_topk_for_users(
        train,
        user_ids,
        seen_items,
        num_items,
        max(ks),
        batch_size,
        strategy,
        seed,
    )
    return multi_target_metrics(topk_items, user_ids, relevant_items, ks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/two_tower_1m.pt")
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed-1m")
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
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
    if max_k > model.num_items:
        raise ValueError("requested K is larger than the movie catalog")

    history = train
    if args.split == "test":
        validation = pd.read_csv(args.processed_dir / "val.csv")
        history = pd.concat((train, validation), ignore_index=True)
    seen_items = build_seen_items(history)
    results = {
        "two_tower": evaluate_retrieval_metrics(
            model,
            evaluation_data,
            seen_items,
            args.ks,
            args.batch_size,
            device,
        )
    }
    if not args.no_baselines:
        for strategy in ("random", "popularity"):
            results[strategy] = evaluate_baseline_metrics(
                train,
                evaluation_data,
                seen_items,
                model.num_items,
                args.ks,
                args.batch_size,
                strategy,
                args.seed,
            )

    print(
        f"device: {device} | split: {args.split} "
        f"| users: {evaluation_data['user_idx'].nunique():,} "
        f"| catalog: {model.num_items:,}"
    )
    for method, metrics in results.items():
        print(f"\n{method}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
