import unittest

import torch

from evaluation.evaluate import single_target_metrics


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

if __name__ == "__main__":
    unittest.main()
