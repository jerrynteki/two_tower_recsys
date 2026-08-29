"""Full-catalog retrieval and filtering utilities."""

from __future__ import annotations

import pandas as pd
import torch

from models import TwoTower


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
