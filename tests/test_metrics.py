import unittest
from math import log2

import torch

from evaluation.evaluate import multi_target_metrics, single_target_metrics


class RetrievalMetricsTest(unittest.TestCase):
    def test_single_target_metrics(self) -> None:
        topk_items = torch.tensor([[2, 3, 4], [5, 6, 7]])
        target_items = torch.tensor([2, 7])

        metrics = single_target_metrics(topk_items, target_items, ks=[1, 3])

        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["HitRate@1"], 0.5)
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertEqual(metrics["HitRate@3"], 1.0)
        self.assertAlmostEqual(metrics["MRR@1"], 0.5)
        self.assertAlmostEqual(metrics["NDCG@1"], 0.5)
        self.assertAlmostEqual(metrics["MRR@3"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics["NDCG@3"], 0.75)

    def test_metrics_reject_unavailable_k(self) -> None:
        with self.assertRaises(ValueError):
            single_target_metrics(torch.tensor([[1, 2]]), torch.tensor([1]), ks=[3])

    def test_multi_target_metrics(self) -> None:
        topk_items = torch.tensor([[2, 3, 4], [5, 6, 7]])
        user_ids = torch.tensor([0, 1])
        relevant_items = {0: {2, 4}, 1: {7, 8}}

        metrics = multi_target_metrics(
            topk_items, user_ids, relevant_items, ks=[1, 3]
        )

        self.assertAlmostEqual(metrics["Recall@1"], 0.25)
        self.assertAlmostEqual(metrics["Recall@3"], 0.75)
        self.assertAlmostEqual(metrics["NDCG@1"], 0.5)
        ideal_dcg = 1.0 + 1.0 / log2(3.0)
        expected_ndcg = (
            (1.0 + 1.0 / log2(4.0)) / ideal_dcg
            + (1.0 / log2(4.0)) / ideal_dcg
        ) / 2.0
        self.assertAlmostEqual(metrics["NDCG@3"], expected_ndcg)

if __name__ == "__main__":
    unittest.main()
