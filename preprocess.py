"""Prepare MovieLens 100K or 1M interactions for two-tower retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/u.data")
OUTPUT_DIR = Path("data/processed")


def load_data(path: Path = RAW_PATH, dataset: str = "100k") -> pd.DataFrame:
    if dataset not in {"100k", "1m"}:
        raise ValueError("dataset must be '100k' or '1m'")
    return pd.read_csv(
        path,
        sep="\t" if dataset == "100k" else "::",
        engine="c" if dataset == "100k" else "python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("100k", "1m"), default="100k")
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-positive-rating", type=int, default=4)
    parser.add_argument("--max-negative-rating", type=int, default=2)
    return parser.parse_args()


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


def build_observed_negative_interactions(
    ratings: pd.DataFrame,
    train: pd.DataFrame,
    user2idx: dict[int, int],
    movie2idx: dict[int, int],
    max_rating: int = 2,
) -> pd.DataFrame:
    """Keep strong dislikes observed before each user's validation period."""
    negatives = ratings.loc[ratings["rating"] <= max_rating].copy()
    negatives["user_idx"] = negatives["user_id"].map(user2idx)
    negatives["movie_idx"] = negatives["movie_id"].map(movie2idx)
    negatives = negatives.dropna(subset=["user_idx", "movie_idx"])
    negatives[["user_idx", "movie_idx"]] = negatives[
        ["user_idx", "movie_idx"]
    ].astype("int64")

    train_cutoffs = (
        train.groupby("user_idx", as_index=False)["timestamp"]
        .max()
        .rename(columns={"timestamp": "train_cutoff"})
    )
    negatives = negatives.merge(train_cutoffs, on="user_idx", how="inner")
    negatives = negatives.loc[
        negatives["timestamp"] <= negatives["train_cutoff"]
    ].drop(columns="train_cutoff")
    return negatives.sort_values(
        ["user_idx", "timestamp", "movie_idx"], kind="stable"
    ).reset_index(drop=True)


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
    observed_negatives: pd.DataFrame,
    user2idx: dict[int, int],
    movie2idx: dict[int, int],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    observed_negatives.to_csv(output_dir / "train_negatives.csv", index=False)
    for filename, mapping in (
        ("user2idx.json", user2idx),
        ("movie2idx.json", movie2idx),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in mapping.items()}, handle, indent=2)


def main() -> None:
    args = parse_args()
    raw_path = args.raw_path or (
        RAW_PATH if args.dataset == "100k" else Path("data/raw/ml-1m/ratings.dat")
    )
    output_dir = args.output_dir or (
        OUTPUT_DIR if args.dataset == "100k" else Path("data/processed-1m")
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Download MovieLens {args.dataset} and place "
            "the ratings file there."
        )
    ratings = load_data(raw_path, args.dataset)
    interactions = filter_positive_interactions(
        ratings, args.min_positive_rating
    )
    interactions, user2idx, movie2idx = build_id_mapping(interactions)
    train, val, test = chronological_split(interactions)
    validate_split(train, val, test)
    observed_negatives = build_observed_negative_interactions(
        ratings,
        train,
        user2idx,
        movie2idx,
        args.max_negative_rating,
    )
    save_processed_data(
        train,
        val,
        test,
        observed_negatives,
        user2idx,
        movie2idx,
        output_dir,
    )

    print(f"Preprocessing complete | dataset: MovieLens {args.dataset}")
    print(f"users: {len(user2idx):,} | movies: {len(movie2idx):,}")
    print(f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"observed train negatives: {len(observed_negatives):,}")
    print(f"artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
