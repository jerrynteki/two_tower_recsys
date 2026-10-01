"""Uniform catalog-negative sampling and sampled-softmax training."""

from __future__ import annotations

import torch
from torch import nn

class UniformNegativeSampler:
    """Sample unseen catalog items uniformly without replacement."""

    def __init__(
        self,
        num_items: int,
        seen_items: dict[int, set[int]],
        seed: int = 42,
    ) -> None:
        self.num_items = num_items
        self.seen_items = seen_items
        self.weights = torch.ones(num_items)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, user_ids: torch.Tensor, count: int) -> torch.Tensor:
        rows = []
        for user_id in user_ids.cpu().tolist():
            weights = self.weights.clone()
            seen = self.seen_items.get(int(user_id), set())
            if seen:
                weights[list(seen)] = 0
            if int((weights > 0).sum()) < count:
                raise ValueError("not enough unseen items to sample without replacement")
            sampled = torch.multinomial(
                weights,
                count,
                replacement=False,
                generator=self.generator,
            )
            rows.append(sampled)
        return torch.stack(rows).to(user_ids.device)


def sampled_logits(
    model: nn.Module,
    user_ids: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
) -> torch.Tensor:
    users = model.user_tower(user_ids)
    positives = model.item_tower(positive_ids)
    negatives = model.item_tower(negative_ids)
    positive_scores = model.score_pairs(users, positives).unsqueeze(1)
    negative_scores = model.score_pairs(users.unsqueeze(1), negatives)
    return torch.cat((positive_scores, negative_scores), dim=1) / model.temperature


def train_sampled_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    sampler: UniformNegativeSampler,
    negative_count: int,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = total_examples = 0
    for user_ids, positive_ids in loader:
        user_ids, positive_ids = user_ids.to(device), positive_ids.to(device)
        negatives = sampler.sample(user_ids, negative_count)
        optimizer.zero_grad()
        logits = sampled_logits(model, user_ids, positive_ids, negatives)
        loss = loss_fn(logits, torch.zeros(len(user_ids), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(user_ids)
        total_examples += len(user_ids)
    return total_loss / total_examples
