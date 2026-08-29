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

if __name__ == "__main__":
    unittest.main()
