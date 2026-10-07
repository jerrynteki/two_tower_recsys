"""Two-tower model whose item tower uses movie metadata."""

import torch
from torch import nn
from torch.nn import functional as F

from models.two_tower import UserTower


class ContentItemTower(nn.Module):
    def __init__(self, feature_matrix: torch.Tensor, embedding_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("feature_matrix", feature_matrix.float())
        self.mlp = nn.Sequential(
            nn.Linear(feature_matrix.shape[1], embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.feature_matrix[item_ids]), p=2, dim=-1)

    @property
    def num_items(self) -> int:
        return self.feature_matrix.shape[0]


class HybridItemTower(nn.Module):
    """Combine trainable item-ID embeddings with projected metadata features."""

    def __init__(
        self,
        feature_matrix: torch.Tensor,
        embedding_dim: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_matrix", feature_matrix.float())
        self.normalize_embeddings = normalize_embeddings
        self.id_embedding = nn.Embedding(len(feature_matrix), embedding_dim)
        self.content_mlp = nn.Sequential(
            nn.Linear(feature_matrix.shape[1], embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    @property
    def num_items(self) -> int:
        return self.feature_matrix.shape[0]

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_vectors = self.id_embedding(item_ids)
        content_vectors = self.content_mlp(self.feature_matrix[item_ids])
        if self.normalize_embeddings:
            id_vectors = F.normalize(id_vectors, p=2, dim=-1)
            content_vectors = F.normalize(content_vectors, p=2, dim=-1)
            return F.normalize(id_vectors + content_vectors, p=2, dim=-1)
        return id_vectors + content_vectors


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

    def score_pairs(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        return (users * items).sum(dim=-1)


class HybridFeatureTwoTower(nn.Module):
    """Two-tower retrieval model combining user/item IDs and movie metadata."""

    def __init__(
        self,
        num_users: int,
        feature_matrix: torch.Tensor,
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
            num_users, embedding_dim, normalize_embeddings, architecture="mlp"
        )
        self.item_tower = HybridItemTower(
            feature_matrix, embedding_dim, normalize_embeddings
        )
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.similarity = similarity
        self.architecture = "feature_hybrid"

    @property
    def num_items(self) -> int:
        return self.item_tower.num_items

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)

    def score_embeddings(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.similarity == "cosine":
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        return user_embeddings @ item_embeddings.T

    def score_pairs(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.similarity == "cosine":
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        return (user_embeddings * item_embeddings).sum(dim=-1)
