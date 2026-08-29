import unittest

import torch

from training.negative_sampling import CatalogNegativeSampler, sampled_logits
from models import TwoTower


class NegativeSamplingTests(unittest.TestCase):
    def test_sampler_excludes_seen_items(self):
        sampler = CatalogNegativeSampler(8, {0: {0, 1, 2}}, seed=1)
        sampled = sampler.sample(torch.tensor([0]), 4)
        self.assertTrue(set(sampled[0].tolist()).isdisjoint({0, 1, 2}))
        self.assertEqual(len(set(sampled[0].tolist())), 4)

    def test_sampled_logits_put_positive_first(self):
        model = TwoTower(2, 6, embedding_dim=4)
        logits = sampled_logits(model, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3, 4], [4, 5]]))
        self.assertEqual(tuple(logits.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
