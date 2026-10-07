"""Load aligned numeric movie features for feature-aware recommender models."""

from pathlib import Path

import pandas as pd
import torch

from features.prepare_movie_features import GENRES


def load_movie_feature_matrix(processed_dir: Path, num_items: int) -> torch.Tensor:
    """Read movie features in movie_idx order and validate catalog alignment."""
    feature_path = processed_dir / "movie_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"movie features not found at {feature_path}; create them with "
            "`python -m features.prepare_movie_features`"
        )

    frame = pd.read_csv(feature_path)
    required_columns = ["movie_idx", "release_year_scaled", *GENRES]
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"movie feature file is missing columns: {', '.join(missing)}")
    frame = frame.sort_values("movie_idx").reset_index(drop=True)
    expected_indices = pd.Series(range(num_items), name="movie_idx")
    if len(frame) != num_items or not frame["movie_idx"].reset_index(drop=True).equals(
        expected_indices
    ):
        raise ValueError(
            "movie_features.csv must contain exactly one row per catalog item, "
            "with contiguous movie_idx values from 0 to num_items - 1"
        )

    values = frame[["release_year_scaled", *GENRES]].to_numpy(dtype="float32")
    return torch.from_numpy(values)
