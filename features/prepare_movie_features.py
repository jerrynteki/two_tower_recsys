"""Convert MovieLens 1M movie metadata into model-ready numeric features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def prepare(input_path: Path, mapping_path: Path, output_path: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        input_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = {int(key): value for key, value in json.load(handle).items()}
    movies = movies[movies.movie_id.isin(mapping)].copy()
    missing_movie_ids = set(mapping) - set(movies.movie_id)
    if missing_movie_ids:
        raise ValueError(
            f"movies.dat is missing {len(missing_movie_ids)} mapped movie IDs"
        )
    movies["movie_idx"] = movies.movie_id.map(mapping)
    year = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False),
        errors="coerce",
    ).fillna(1995)
    movies["release_year_scaled"] = (year - 1900) / 100
    genre_indicators = movies["genres"].str.get_dummies(sep="|").reindex(
        columns=GENRES, fill_value=0
    )
    movies[GENRES] = genre_indicators.astype("int64")
    result = movies[
        ["movie_idx", "movie_id", "title", "release_year_scaled", *GENRES]
    ].sort_values("movie_idx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/ml-1m/movies.dat")
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/processed-1m/movie2idx.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed-1m/movie_features.csv")
    )
    args = parser.parse_args()
    result = prepare(args.input, args.mapping, args.output)
    print(f"movie features: {len(result):,} rows x {len(GENRES) + 1} features")
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
