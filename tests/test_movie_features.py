import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from features.prepare_movie_features import prepare


class MovieFeatureTests(unittest.TestCase):
    def test_prepare_supports_movielens_1m_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "movies.dat"
            mapping_path = root / "movie2idx.json"
            output_path = root / "movie_features.csv"
            input_path.write_text(
                "1::Toy Story (1995)::Animation|Children's|Comedy\n"
                "2::Jumanji (1995)::Adventure|Children's|Fantasy\n",
                encoding="latin-1",
            )
            mapping_path.write_text(json.dumps({"1": 0, "2": 1}))

            result = prepare(input_path, mapping_path, output_path)

            self.assertEqual(result["movie_idx"].tolist(), [0, 1])
            self.assertEqual(result["release_year_scaled"].tolist(), [0.95, 0.95])
            self.assertEqual(result["Animation"].tolist(), [1, 0])
            self.assertEqual(result["Fantasy"].tolist(), [0, 1])
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
