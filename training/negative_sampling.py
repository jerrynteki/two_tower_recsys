"""Catalog negative samplers and sampled-softmax training."""

from __future__ import annotations

import torch
from torch import nn

from models import TwoTower


class CatalogNegativeSampler:
    """Sample catalog negatives, optionally mixing in observed dislikes."""

    def __init__(
        self,
        num_items: int,
        seen_items: dict[int, set[int]],
        popularity: torch.Tensor | None = None,
        observed_negatives: dict[int, set[int]] | None = None,
        observed_fraction: float = 0.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= observed_fraction <= 1.0:
            raise ValueError("observed_fraction must be between 0 and 1")
        self.num_items = num_items
        self.seen_items = seen_items
        self.weights = torch.ones(num_items) if popularity is None else popularity.float().clamp_min(1)
        self.observed_negatives = observed_negatives or {}
        self.observed_fraction = observed_fraction
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, user_ids: torch.Tensor, count: int, candidate_count: int | None = None) -> torch.Tensor:
        width = candidate_count or count
        rows = []
        for user_id in user_ids.cpu().tolist():
            weights = self.weights.clone()
            seen = self.seen_items.get(int(user_id), set())
            if seen:
                weights[list(seen)] = 0

            observed = sorted(
                self.observed_negatives.get(int(user_id), set()) - seen
            )
            observed_count = min(
                len(observed), round(width * self.observed_fraction)
            )
            chosen_observed = torch.empty(0, dtype=torch.long)
            if observed_count:
                observed_tensor = torch.tensor(observed, dtype=torch.long)
                order = torch.randperm(
                    len(observed_tensor), generator=self.generator
                )[:observed_count]
                chosen_observed = observed_tensor[order]
                weights[observed_tensor] = 0

            catalog_count = width - observed_count
            if int((weights > 0).sum()) < catalog_count:
                raise ValueError("not enough unseen items to sample without replacement")
            sampled_catalog = torch.multinomial(
                weights,
                catalog_count,
                replacement=False,
                generator=self.generator,
            )
            rows.append(torch.cat((chosen_observed, sampled_catalog)))
        return torch.stack(rows).to(user_ids.device)


def build_item_sets(interactions) -> dict[int, set[int]]:
    """Group an interaction table into user-to-item sets."""
    return {
        int(user_id): set(group["movie_idx"].astype(int))
        for user_id, group in interactions.groupby("user_idx")
    }


def sampled_logits(model: TwoTower, user_ids: torch.Tensor, positive_ids: torch.Tensor, negative_ids: torch.Tensor) -> torch.Tensor:
    users = model.user_tower(user_ids)
    positives = model.item_tower(positive_ids)
    negatives = model.item_tower(negative_ids)
    positive_scores = model.score_pairs(users, positives).unsqueeze(1)
    negative_scores = model.score_pairs(users.unsqueeze(1), negatives)
    return torch.cat((positive_scores, negative_scores), dim=1) / model.temperature


def train_sampled_epoch(model: TwoTower, loader, optimizer, device: torch.device, sampler: CatalogNegativeSampler, negative_count: int, hard_pool_size: int | None = None) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = total_examples = 0
    for user_ids, positive_ids in loader:
        user_ids, positive_ids = user_ids.to(device), positive_ids.to(device)
        negatives = sampler.sample(user_ids, negative_count, hard_pool_size)
        if hard_pool_size:
            with torch.no_grad():
                users = model.user_tower(user_ids).unsqueeze(1)
                candidates = model.item_tower(negatives)
                scores = model.score_pairs(users, candidates)
                chosen = scores.topk(negative_count, dim=1).indices
                negatives = negatives.gather(1, chosen)
        optimizer.zero_grad()
        logits = sampled_logits(model, user_ids, positive_ids, negatives)
        loss = loss_fn(logits, torch.zeros(len(user_ids), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(user_ids)
        total_examples += len(user_ids)
    return total_loss / total_examples
