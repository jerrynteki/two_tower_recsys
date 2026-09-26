import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preprocess import build_observed_negative_interactions, load_data


class PreprocessTests(unittest.TestCase):
    def test_load_data_supports_movielens_1m_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.dat"
            path.write_text(
                "1::1193::5::978300760\n2::661::3::978302109\n",
                encoding="utf-8",
            )

            result = load_data(path, dataset="1m")

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


if __name__ == "__main__":
    unittest.main()
