import unittest

import torch

from models.feature_two_tower import ContentItemTower, FeatureTwoTower
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


if __name__ == "__main__":
    unittest.main()
