"""LightGCN collaborative-filtering model for implicit user-item graphs."""

from __future__ import annotations

import torch
from torch import nn


def normalized_bipartite_adjacency(
    num_users: int,
    num_items: int,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Build the symmetrically normalized user-item graph from train edges."""
    user_ids = user_ids.to(dtype=torch.long, device="cpu")
    item_nodes = item_ids.to(dtype=torch.long, device="cpu") + num_users
    nodes = torch.cat((user_ids, item_nodes))
    neighbors = torch.cat((item_nodes, user_ids))
    indices = torch.stack((nodes, neighbors))
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (num_users + num_items, num_users + num_items),
        check_invariants=False,
    ).coalesce()
    degree = torch.zeros(num_users + num_items, dtype=torch.float32)
    degree.index_add_(0, adjacency.indices()[0], adjacency.values())
    inv_sqrt_degree = degree.clamp_min_(1).pow_(-0.5)
    row, col = adjacency.indices()
    normalized_values = adjacency.values() * inv_sqrt_degree[row] * inv_sqrt_degree[col]
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        normalized_values,
        adjacency.shape,
        check_invariants=False,
    ).coalesce()


class LightGCN(nn.Module):
    """Propagate ID embeddings over the train interaction graph, then average layers."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        adjacency: torch.Tensor,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_layers < 0:
            raise ValueError("embedding_dim must be positive and num_layers nonnegative")
        if adjacency.shape != (num_users + num_items, num_users + num_items):
            raise ValueError("adjacency shape must match the user and item counts")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("adjacency", adjacency.coalesce())

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        layer_embeddings = [ego_embeddings]
        current = ego_embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self.adjacency, current)
            layer_embeddings.append(current)
        final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        regularization: float,
    ) -> torch.Tensor:
        users, items = self.propagated_embeddings()
        user_vectors = users[user_ids]
        positive_vectors = items[positive_ids]
        negative_vectors = items[negative_ids]
        positive_scores = (user_vectors * positive_vectors).sum(dim=-1)
        negative_scores = (user_vectors * negative_vectors).sum(dim=-1)
        ranking_loss = -torch.nn.functional.logsigmoid(
            positive_scores - negative_scores
        ).mean()
        ego_user = self.user_embedding(user_ids)
        ego_positive = self.item_embedding(positive_ids)
        ego_negative = self.item_embedding(negative_ids)
        regularization_loss = (
            ego_user.square().sum(dim=-1)
            + ego_positive.square().sum(dim=-1)
            + ego_negative.square().sum(dim=-1)
        ).mean() / 2
        return ranking_loss + regularization * regularization_loss
