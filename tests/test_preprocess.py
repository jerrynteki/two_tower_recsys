import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preprocess import (
    build_observed_negative_interactions,
    filter_k_core,
    flowcf_split,
    load_data,
    validate_flowcf_split,
)


class PreprocessTests(unittest.TestCase):
    def test_load_data_supports_movielens_1m_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.dat"
            path.write_text(
                "1::1193::5::978300760\n2::661::3::978302109\n",
                encoding="utf-8",
            )

            result = load_data(path)

        self.assertEqual(result.columns.tolist(), [
            "user_id", "movie_id", "rating", "timestamp"
        ])
        self.assertEqual(result.iloc[0].to_dict(), {
            "user_id": 1,
            "movie_id": 1193,
            "rating": 5,
            "timestamp": 978300760,
        })

    def test_observed_negatives_are_strong_dislikes_before_cutoff(self) -> None:
        ratings = pd.DataFrame(
            {
                "user_id": [10, 10, 10, 20],
                "movie_id": [100, 101, 102, 100],
                "rating": [1, 2, 1, 2],
                "timestamp": [5, 15, 25, 5],
            }
        )
        train = pd.DataFrame(
            {"user_idx": [0], "movie_idx": [0], "timestamp": [20]}
        )

        result = build_observed_negative_interactions(
            ratings,
            train,
            user2idx={10: 0},
            movie2idx={100: 0, 101: 1, 102: 2},
        )

        self.assertEqual(result["movie_idx"].tolist(), [0, 1])
        self.assertTrue((result["timestamp"] <= 20).all())

    def test_k_core_filter_repeats_until_users_and_movies_are_eligible(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 3, 3],
                "movie_id": [10, 11, 12, 10, 11, 10, 13],
            }
        )

        result = filter_k_core(interactions, min_interactions=2)

        self.assertEqual(set(result["user_id"]), {1, 2})
        self.assertEqual(set(result["movie_id"]), {10, 11})
        self.assertTrue((result.groupby("user_id").size() >= 2).all())
        self.assertTrue((result.groupby("movie_id").size() >= 2).all())

    def test_flowcf_split_matches_recbole_rounding_and_is_reproducible(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_idx": [0] * 10 + [1] * 5,
                "movie_idx": list(range(10)) + list(range(10, 15)),
            }
        )

        first = flowcf_split(interactions, seed=2020)
        second = flowcf_split(interactions, seed=2020)

        self.assertEqual([len(part) for part in first], [11, 2, 2])
        for first_part, second_part in zip(first, second):
            pd.testing.assert_frame_equal(first_part, second_part)
        split_pairs = [
            set(zip(part["user_idx"], part["movie_idx"])) for part in first
        ]
        self.assertFalse(split_pairs[0] & split_pairs[1])
        self.assertFalse(split_pairs[0] & split_pairs[2])
        self.assertFalse(split_pairs[1] & split_pairs[2])
        self.assertEqual(len(set.union(*split_pairs)), len(interactions))

    def test_flowcf_split_validation_checks_membership_and_user_sizes(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1] * 10 + [2] * 5,
                "movie_id": list(range(10)) + list(range(10, 15)),
                "user_idx": [0] * 10 + [1] * 5,
                "movie_idx": list(range(15)),
            }
        )

        train, val, test = flowcf_split(interactions, seed=2020)

        validate_flowcf_split(interactions, train, val, test)


if __name__ == "__main__":
    unittest.main()
