"""Prepare MovieLens 100K interactions for two-tower retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/u.data")
OUTPUT_DIR = Path("data/processed")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
    )


def filter_positive_interactions(
    df: pd.DataFrame, min_rating: int = 4
) -> pd.DataFrame:
    """Treat ratings >= min_rating as positive implicit feedback."""
    return df.loc[df["rating"] >= min_rating].copy()


def build_id_mapping(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Map sparse raw IDs to contiguous embedding-table indices."""
    df = df.copy()
    user_ids = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())
    user2idx = {int(raw_id): idx for idx, raw_id in enumerate(user_ids)}
    movie2idx = {int(raw_id): idx for idx, raw_id in enumerate(movie_ids)}
    df["user_idx"] = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)
    return df, user2idx, movie2idx


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep each eligible user's last two positives for validation and test."""
    ordered = df.sort_values(
        ["user_idx", "timestamp", "movie_idx"], kind="stable"
    )
    eligible = ordered.groupby("user_idx")["user_idx"].transform("size") >= 3
    ordered = ordered.loc[eligible]

    position_from_end = ordered.groupby("user_idx").cumcount(ascending=False)
    train = ordered.loc[position_from_end >= 2].reset_index(drop=True)
    val = ordered.loc[position_from_end == 1].reset_index(drop=True)
    test = ordered.loc[position_from_end == 0].reset_index(drop=True)
    return train, val, test


def validate_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    assert not train.empty and not val.empty and not test.empty
    assert val["user_idx"].is_unique
    assert test["user_idx"].is_unique
    assert set(val["user_idx"]) == set(test["user_idx"])
    assert set(val["user_idx"]).issubset(set(train["user_idx"]))

    boundaries = (
        train.groupby("user_idx")["timestamp"].max().rename("train_max")
        .to_frame()
        .join(val.set_index("user_idx")["timestamp"].rename("val_ts"))
        .join(test.set_index("user_idx")["timestamp"].rename("test_ts"))
    )
    # MovieLens timestamps have one-second precision, so simultaneous ratings
    # can tie. Stable ordering prevents leakage while allowing equal timestamps.
    assert (boundaries["train_max"] <= boundaries["val_ts"]).all()
    assert (boundaries["val_ts"] <= boundaries["test_ts"]).all()


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user2idx: dict[int, int],
    movie2idx: dict[int, int],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    for filename, mapping in (
        ("user2idx.json", user2idx),
        ("movie2idx.json", movie2idx),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in mapping.items()}, handle, indent=2)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Download MovieLens 100K and copy u.data there."
        )
    interactions = filter_positive_interactions(load_data())
    interactions, user2idx, movie2idx = build_id_mapping(interactions)
    train, val, test = chronological_split(interactions)
    validate_split(train, val, test)
    save_processed_data(train, val, test, user2idx, movie2idx)

    print("Day 1 preprocessing complete")
    print(f"users: {len(user2idx):,} | movies: {len(movie2idx):,}")
    print(f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"artifacts: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

