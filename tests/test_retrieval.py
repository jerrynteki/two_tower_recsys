import unittest

import pandas as pd
import torch

from evaluation.evaluate import (
    build_seen_items,
    mask_seen_items,
    retrieve_baseline_topk,
)


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

    def test_popularity_baseline_ranks_frequent_unseen_items(self) -> None:
        train = pd.DataFrame(
            {
                "user_idx": [0, 1, 2, 3, 1, 2],
                "movie_idx": [0, 0, 0, 1, 1, 2],
            }
        )
        evaluation = pd.DataFrame(
            {"user_idx": [10, 20], "movie_idx": [1, 0]}
        )
        seen = {10: {0}, 20: {1}}

        topk, targets = retrieve_baseline_topk(
            train,
            evaluation,
            seen,
            num_items=4,
            max_k=2,
            batch_size=2,
            strategy="popularity",
        )

        torch.testing.assert_close(topk, torch.tensor([[1, 2], [0, 2]]))
        torch.testing.assert_close(targets, torch.tensor([1, 0]))

    def test_random_baseline_is_seeded_and_masks_seen_items(self) -> None:
        train = pd.DataFrame({"user_idx": [0], "movie_idx": [0]})
        evaluation = pd.DataFrame(
            {"user_idx": [10, 20], "movie_idx": [1, 2]}
        )
        seen = {10: {0}, 20: {3}}

        first, _ = retrieve_baseline_topk(
            train, evaluation, seen, 4, 2, 2, "random", seed=7
        )
        second, _ = retrieve_baseline_topk(
            train, evaluation, seen, 4, 2, 2, "random", seed=7
        )

        torch.testing.assert_close(first, second)
        self.assertNotIn(0, first[0].tolist())
        self.assertNotIn(3, first[1].tolist())


if __name__ == "__main__":
    unittest.main()
