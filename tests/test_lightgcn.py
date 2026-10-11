import unittest

import torch

from models.lightgcn import LightGCN, normalized_bipartite_adjacency
from training.train_lightgcn import sample_negatives


class LightGCNTests(unittest.TestCase):
    def test_propagated_embeddings_have_expected_shape_and_gradients(self):
        adjacency = normalized_bipartite_adjacency(
            2,
            3,
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 2]),
        )
        model = LightGCN(2, 3, adjacency, embedding_dim=4, num_layers=2)
        users, items = model.propagated_embeddings()
        self.assertEqual(users.shape, (2, 4))
        self.assertEqual(items.shape, (3, 4))
        loss = model.bpr_loss(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([2]), 1e-4
        )
        loss.backward()
        self.assertIsNotNone(model.user_embedding.weight.grad)
        self.assertIsNotNone(model.item_embedding.weight.grad)

    def test_negative_sampler_excludes_training_interactions(self):
        seen = torch.tensor([[True, False, True], [False, True, False]])
        generator = torch.Generator().manual_seed(7)
        users = torch.tensor([0, 1, 0, 1])
        negatives = sample_negatives(users, seen, 3, generator)
        self.assertFalse(seen[users, negatives].any())


if __name__ == "__main__":
    unittest.main()
