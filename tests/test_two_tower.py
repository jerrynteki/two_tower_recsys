import unittest

import torch
from torch.nn import functional as F

from models import TwoTower


class TwoTowerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = TwoTower(num_users=5, num_items=7, embedding_dim=8)

    def test_towers_return_normalized_vectors(self) -> None:
        users, items = self.model(torch.tensor([0, 1]), torch.tensor([2, 3]))
        self.assertEqual(users.shape, (2, 8))
        self.assertEqual(items.shape, (2, 8))
        self.assertTrue(torch.allclose(users.norm(dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(items.norm(dim=1), torch.ones(2)))

    def test_in_batch_objective_has_one_class_per_item(self) -> None:
        logits = self.model.in_batch_logits(
            torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
        )
        labels = torch.arange(3)
        self.assertEqual(logits.shape, (3, 3))
        self.assertTrue(torch.isfinite(F.cross_entropy(logits, labels)))

    def test_cosine_scoring_normalizes_raw_tower_outputs(self) -> None:
        model = TwoTower(
            num_users=5,
            num_items=7,
            embedding_dim=8,
            normalize_embeddings=False,
            similarity="cosine",
        )
        users, items = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
        actual = model.score_embeddings(users, items)
        expected = F.normalize(users, dim=-1) @ F.normalize(items, dim=-1).T
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
