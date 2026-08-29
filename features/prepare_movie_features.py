"""Convert MovieLens u.item metadata into model-ready numeric features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def prepare(input_path: Path, mapping_path: Path, output_path: Path) -> pd.DataFrame:
    columns = ["movie_id", "title", "release_date", "video_release_date", "imdb_url", *GENRES]
    movies = pd.read_csv(input_path, sep="|", names=columns, encoding="latin-1")
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = {int(key): value for key, value in json.load(handle).items()}
    movies = movies[movies.movie_id.isin(mapping)].copy()
    movies["movie_idx"] = movies.movie_id.map(mapping)
    year = pd.to_datetime(movies.release_date, errors="coerce").dt.year.fillna(1995)
    movies["release_year_scaled"] = (year - 1900) / 100
    result = movies[["movie_idx", "movie_id", "title", "release_year_scaled", *GENRES]].sort_values("movie_idx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/u.item"))
    parser.add_argument("--mapping", type=Path, default=Path("data/processed/movie2idx.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/movie_features.csv"))
    args = parser.parse_args()
    result = prepare(args.input, args.mapping, args.output)
    print(f"movie features: {len(result):,} rows x {len(GENRES) + 1} features")
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
