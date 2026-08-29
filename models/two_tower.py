"""ID-based two-tower retrieval model."""

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingTower(nn.Module):
    """Convert a categorical ID into a normalized retrieval embedding."""

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.mlp(self.embedding(entity_ids))
        if self.normalize_embeddings:
            return F.normalize(vectors, p=2, dim=-1)
        return vectors


class UserTower(EmbeddingTower):
    pass


class ItemTower(EmbeddingTower):
    pass


class TwoTower(nn.Module):
    """Encode users and items independently into the same vector space."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
        similarity: str = "dot",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity not in {"dot", "cosine"}:
            raise ValueError("similarity must be 'dot' or 'cosine'")
        self.user_tower = UserTower(
            num_users, embedding_dim, normalize_embeddings
        )
        self.item_tower = ItemTower(
            num_items, embedding_dim, normalize_embeddings
        )
        self.temperature = temperature
        self.similarity = similarity

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)

    def in_batch_logits(
        self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score every user against every positive item in the batch."""
        user_embeddings, item_embeddings = self(user_ids, positive_item_ids)
        return self.score_embeddings(user_embeddings, item_embeddings) / self.temperature

    def score_embeddings(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return every user-to-item similarity score."""
        if self.similarity == "cosine":
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        return user_embeddings @ item_embeddings.T

    def score_pairs(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score aligned user-item pairs, including broadcast negative sets."""
        if self.similarity == "cosine":
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        return (user_embeddings * item_embeddings).sum(dim=-1)
