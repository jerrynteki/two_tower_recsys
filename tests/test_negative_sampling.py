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

    def test_hybrid_sampler_includes_observed_dislikes(self):
        sampler = CatalogNegativeSampler(
            10,
            {0: {0}},
            observed_negatives={0: {5, 6, 7}},
            observed_fraction=0.5,
            seed=1,
        )

        sampled = sampler.sample(torch.tensor([0]), 4)

        observed_count = len(set(sampled[0].tolist()) & {5, 6, 7})
        self.assertEqual(observed_count, 2)
        self.assertNotIn(0, sampled[0].tolist())


if __name__ == "__main__":
    unittest.main()
