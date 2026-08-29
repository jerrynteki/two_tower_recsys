"""Catalog negative samplers and sampled-softmax training."""

from __future__ import annotations

import torch
from torch import nn

from models import TwoTower


class CatalogNegativeSampler:
    """Sample unseen catalog items uniformly or by item popularity."""

    def __init__(self, num_items: int, seen_items: dict[int, set[int]], popularity: torch.Tensor | None = None, seed: int = 42) -> None:
        self.num_items = num_items
        self.seen_items = seen_items
        self.weights = torch.ones(num_items) if popularity is None else popularity.float().clamp_min(1)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, user_ids: torch.Tensor, count: int, candidate_count: int | None = None) -> torch.Tensor:
        width = candidate_count or count
        rows = []
        for user_id in user_ids.cpu().tolist():
            weights = self.weights.clone()
            seen = self.seen_items.get(int(user_id), set())
            if seen:
                weights[list(seen)] = 0
            if int((weights > 0).sum()) < width:
                raise ValueError("not enough unseen items to sample without replacement")
            rows.append(torch.multinomial(weights, width, replacement=False, generator=self.generator))
        return torch.stack(rows).to(user_ids.device)


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
