"""Metrics for single-target Top-K retrieval evaluation."""

from collections.abc import Iterable

import torch


def single_target_metrics(
    topk_items: torch.Tensor,
    target_items: torch.Tensor,
    ks: Iterable[int],
) -> dict[str, float]:
    """Calculate Recall@K and HitRate@K for one relevant item per user."""
    if topk_items.ndim != 2:
        raise ValueError("topk_items must have shape [num_users, max_k]")
    if target_items.ndim != 1 or len(target_items) != len(topk_items):
        raise ValueError("target_items must have shape [num_users]")

    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        if k <= 0 or k > topk_items.shape[1]:
            raise ValueError(f"k={k} is outside the available Top-K results")
        hits = (topk_items[:, :k] == target_items[:, None]).any(dim=1)
        hit_rate = hits.float().mean().item()
        # With exactly one held-out relevant item per user, recall and hit rate
        # are numerically identical: each user either retrieves that item or not.
        metrics[f"Recall@{k}"] = hit_rate
        metrics[f"HitRate@{k}"] = hit_rate
    return metrics
