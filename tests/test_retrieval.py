import unittest

import pandas as pd
import torch

from evaluation.retrieval import build_seen_items, mask_seen_items


class RetrievalTest(unittest.TestCase):
    def test_build_seen_items_groups_movies_by_user(self) -> None:
        train = pd.DataFrame(
            {"user_idx": [0, 0, 1], "movie_idx": [2, 3, 4]}
        )

        self.assertEqual(build_seen_items(train), {0: {2, 3}, 1: {4}})

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
