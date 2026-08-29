"""Two-tower model whose item tower uses movie metadata."""

import torch
from torch import nn
from torch.nn import functional as F

from models.two_tower import UserTower


class ContentItemTower(nn.Module):
    def __init__(self, feature_matrix: torch.Tensor, embedding_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("feature_matrix", feature_matrix.float())
        self.mlp = nn.Sequential(nn.Linear(feature_matrix.shape[1], embedding_dim * 2), nn.ReLU(), nn.Linear(embedding_dim * 2, embedding_dim))

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.feature_matrix[item_ids]), p=2, dim=-1)


class FeatureTwoTower(nn.Module):
    def __init__(self, num_users: int, feature_matrix: torch.Tensor, embedding_dim: int = 64, temperature: float = 0.07) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ContentItemTower(feature_matrix, embedding_dim)
        self.temperature = temperature
        self.similarity = "dot"

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        return self.user_tower(user_ids), self.item_tower(item_ids)

    def score_embeddings(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return users @ items.T

    def in_batch_logits(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        users, items = self(user_ids, item_ids)
        return self.score_embeddings(users, items) / self.temperature
