import unittest

import torch

from evaluation.metrics import mask_seen_items, single_target_metrics


class RetrievalMetricsTest(unittest.TestCase):
    def test_single_target_metrics(self) -> None:
        topk_items = torch.tensor([[2, 3, 4], [5, 6, 7]])
        target_items = torch.tensor([2, 7])

        metrics = single_target_metrics(topk_items, target_items, ks=[1, 3])

        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["HitRate@1"], 0.5)
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertEqual(metrics["HitRate@3"], 1.0)

    def test_seen_items_are_removed_from_ranking(self) -> None:
        scores = torch.tensor([[0.1, 0.9, 0.8], [0.7, 0.2, 0.6]])
        user_ids = torch.tensor([10, 20])
        seen_items = {10: {1}, 20: {0, 2}}

        masked = mask_seen_items(scores, user_ids, seen_items)

        self.assertTrue(torch.isneginf(masked[0, 1]))
        self.assertTrue(torch.isneginf(masked[1, 0]))
        self.assertTrue(torch.isneginf(masked[1, 2]))
        self.assertEqual(masked[0, 2].item(), scores[0, 2].item())


if __name__ == "__main__":
    unittest.main()

