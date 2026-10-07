"""Prepare MovieLens 1M interactions for two-tower retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


RAW_PATH = Path("data/raw/ml-1m/ratings.dat")
OUTPUT_DIR = Path("data/processed-1m")
SPLIT_SEED = 2020
EXPECTED_COUNTS = {
    "users": 6_034,
    "movies": 3_125,
    "interactions": 574_376,
}


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the ``ratings.dat`` format distributed with MovieLens 1M."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
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
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="random seed for the per-user 80/10/10 split",
    )
    return parser.parse_args()


def filter_positive_interactions(
    df: pd.DataFrame, min_rating: int = 4
) -> pd.DataFrame:
    """Treat ratings >= min_rating as positive implicit feedback."""
    return df.loc[df["rating"] >= min_rating].copy()


def build_id_mapping(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Map raw IDs to contiguous zero-based indices for embedding tables."""
    df = df.copy()
    user_ids = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())
    user2idx = {int(raw_id): idx for idx, raw_id in enumerate(user_ids)}
    movie2idx = {int(raw_id): idx for idx, raw_id in enumerate(movie_ids)}
    df["user_idx"] = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)
    return df, user2idx, movie2idx


def filter_k_core(
    df: pd.DataFrame, min_interactions: int = 5
) -> pd.DataFrame:
    """Iteratively retain users and movies with enough positive interactions."""
    if min_interactions <= 0:
        raise ValueError("min_interactions must be positive")

    filtered = df.copy()
    while True:
        previous_size = len(filtered)
        user_counts = filtered.groupby("user_id").size()
        movie_counts = filtered.groupby("movie_id").size()
        filtered = filtered.loc[
            filtered["user_id"].isin(
                user_counts[user_counts >= min_interactions].index
            )
            & filtered["movie_id"].isin(
                movie_counts[movie_counts >= min_interactions].index
            )
        ]
        if len(filtered) == previous_size:
            return filtered.copy()


def _ratio_split_counts(total: int) -> tuple[int, int, int]:
    """Return per-user counts for an 80/10/10 split."""
    ratios = (0.8, 0.1, 0.1)
    counts = [int(ratio * total) for ratio in ratios]
    counts[0] = total - counts[1] - counts[2]
    for offset in range(1, len(counts)):
        if counts[0] <= 1:
            break
        ratio = ratios[-offset]
        if 0 < ratio * total < 1:
            counts[-offset] += 1
            counts[0] -= 1
    return counts[0], counts[1], counts[2]


def random_user_split(
    df: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split each user's interactions into 80/10/10 partitions."""
    generator = torch.Generator().manual_seed(seed)
    shuffled = df.iloc[torch.randperm(len(df), generator=generator).numpy()]
    partitions: list[list[pd.DataFrame]] = [[], [], []]

    for _, user_interactions in shuffled.groupby("user_idx", sort=False):
        train_count, val_count, _ = _ratio_split_counts(len(user_interactions))
        train_end = train_count
        val_end = train_end + val_count
        partitions[0].append(user_interactions.iloc[:train_end])
        partitions[1].append(user_interactions.iloc[train_end:val_end])
        partitions[2].append(user_interactions.iloc[val_end:])

    return tuple(
        pd.concat(partition, ignore_index=True) for partition in partitions
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Verify the expected MovieLens 1M 5-core filtering result."""
    actual = {
        "users": int(df["user_id"].nunique()),
        "movies": int(df["movie_id"].nunique()),
        "interactions": len(df),
    }
    if actual != EXPECTED_COUNTS:
        raise ValueError(
            "MovieLens 1M reference counts do not match. Expected "
            f"{EXPECTED_COUNTS}, got {actual}. Check that ratings.dat "
            "is the complete official MovieLens 1M file."
        )
    if (df.groupby("user_id").size() < 5).any():
        raise ValueError("5-core check failed for at least one user")
    if (df.groupby("movie_id").size() < 5).any():
        raise ValueError("5-core check failed for at least one movie")


def validate_random_split(
    source: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify per-user split membership and 80/10/10 sizes."""
    parts = (train, val, test)
    if any(part.empty for part in parts):
        raise ValueError("Split contains an empty partition")

    source_users = set(source["user_idx"].astype(int))
    if any(set(part["user_idx"].astype(int)) != source_users for part in parts):
        raise ValueError("Every user must occur in train, validation, and test")

    key_columns = ["user_id", "movie_id"]
    source_keys = set(map(tuple, source[key_columns].to_numpy()))
    part_keys = [set(map(tuple, part[key_columns].to_numpy())) for part in parts]
    if len(source_keys) != len(source):
        raise ValueError("MovieLens 1M should have unique user-movie pairs")
    if (
        part_keys[0] & part_keys[1]
        or part_keys[0] & part_keys[2]
        or part_keys[1] & part_keys[2]
    ):
        raise ValueError("Split partitions overlap")
    if set.union(*part_keys) != source_keys:
        raise ValueError("Split does not preserve every filtered interaction")

    source_counts = source.groupby("user_idx").size()
    actual_counts = [part.groupby("user_idx").size() for part in parts]
    for user_id, total in source_counts.items():
        expected = _ratio_split_counts(int(total))
        actual = tuple(int(counts.loc[user_id]) for counts in actual_counts)
        if actual != expected:
            raise ValueError(
                f"Split count mismatch for user {user_id}: "
                f"expected {expected}, got {actual}"
            )


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user2idx: dict[int, int],
    movie2idx: dict[int, int],
    output_dir: Path = OUTPUT_DIR,
    metadata: dict[str, object] | None = None,
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
    if metadata is not None:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    raw_path = args.raw_path or RAW_PATH
    output_dir = args.output_dir or OUTPUT_DIR
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Download and extract MovieLens 1M so that "
            "ratings.dat is available at this path."
        )
    ratings = load_data(raw_path)
    interactions = filter_positive_interactions(ratings, min_rating=4)
    interactions = filter_k_core(interactions, min_interactions=5)
    validate_dataset(interactions)
    interactions, user2idx, movie2idx = build_id_mapping(interactions)
    train, val, test = random_user_split(interactions, seed=args.seed)
    validate_random_split(interactions, train, val, test)
    save_processed_data(
        train,
        val,
        test,
        user2idx,
        movie2idx,
        output_dir,
        metadata={
            "dataset": "MovieLens 1M",
            "protocol": "random_user_80_10_10",
            "min_positive_rating": 4,
            "min_interactions": 5,
            "seed": args.seed,
            "split": "random_user_80_10_10",
            "real_item_count": len(movie2idx),
        },
    )

    print(
        f"Preprocessing complete | dataset: MovieLens 1M "
        "| split: random user-level 80/10/10"
    )
    print(f"users: {len(user2idx):,} | movies: {len(movie2idx):,}")
    print(f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
