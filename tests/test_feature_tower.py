import unittest

import torch

from models.feature_two_tower import (
    ContentItemTower,
    FeatureTwoTower,
    GatedHybridItemTower,
)
from training.negative_sampling import sampled_logits


class FeatureTowerTests(unittest.TestCase):
    def test_identical_features_give_identical_embeddings(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tower = ContentItemTower(features, embedding_dim=4)
        vectors = tower(torch.tensor([0, 1, 2]))
        self.assertTrue(torch.allclose(vectors[0], vectors[1]))
        self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(3)))

    def test_feature_model_supports_sampled_training_logits(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
        )
        model = FeatureTwoTower(2, features, embedding_dim=4)

        logits = sampled_logits(
            model,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([[2, 3], [2, 3]]),
        )

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_gated_hybrid_item_tower_starts_at_equal_blend_and_learns_gate(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tower = GatedHybridItemTower(features, embedding_dim=4)
        self.assertAlmostEqual(tower.feature_gate.item(), 0.5)
        output = tower(torch.tensor([0, 1]))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2)))
        output.sum().backward()
        self.assertIsNotNone(tower.feature_gate_logit.grad)


if __name__ == "__main__":
    unittest.main()
