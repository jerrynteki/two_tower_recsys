import unittest

import torch

from models.feature_two_tower import ContentItemTower


class FeatureTowerTests(unittest.TestCase):
    def test_identical_features_give_identical_embeddings(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tower = ContentItemTower(features, embedding_dim=4)
        vectors = tower(torch.tensor([0, 1, 2]))
        self.assertTrue(torch.allclose(vectors[0], vectors[1]))
        self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
