"""Prepare MovieLens 1M interactions for two-tower retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


RAW_PATH = Path("data/raw/ml-1m/ratings.dat")
OUTPUT_DIR = Path("data/processed-1m")
FLOWCF_OUTPUT_DIR = Path("data/processed-1m-flowcf")
FLOWCF_SEED = 2020
FLOWCF_EXPECTED_COUNTS = {
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
    parser.add_argument("--min-positive-rating", type=int, default=4)
    parser.add_argument(
        "--protocol",
        choices=("chronological", "flowcf"),
        default="chronological",
        help="data split and filtering protocol",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FLOWCF_SEED,
        help="random seed for the FlowCF-compatible split",
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
    """Map raw IDs to zero-based indices for embedding tables.

    RecBole reserves index 0 for padding and starts real IDs at 1. This project
    has no padding row, so it uses 0..N-1 for the same real users and movies.
    The relabeling does not change split membership or ranking metrics.
    """
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
    """Match RecBole's per-user 80/10/10 rounding behavior."""
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


def flowcf_split(
    df: pd.DataFrame, seed: int = 2020
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce RecBole's random, user-grouped 80/10/10 split."""
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


def validate_flowcf_dataset(df: pd.DataFrame) -> None:
    """Verify the official FlowCF MovieLens 1M filtering result."""
    actual = {
        "users": int(df["user_id"].nunique()),
        "movies": int(df["movie_id"].nunique()),
        "interactions": len(df),
    }
    if actual != FLOWCF_EXPECTED_COUNTS:
        raise ValueError(
            "FlowCF reference counts do not match. Expected "
            f"{FLOWCF_EXPECTED_COUNTS}, got {actual}. Check that ratings.dat "
            "is the complete official MovieLens 1M file."
        )
    if (df.groupby("user_id").size() < 5).any():
        raise ValueError("FlowCF 5-core check failed for at least one user")
    if (df.groupby("movie_id").size() < 5).any():
        raise ValueError("FlowCF 5-core check failed for at least one movie")


def validate_flowcf_split(
    source: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify RecBole-style user-grouped split membership and sizes."""
    parts = (train, val, test)
    if any(part.empty for part in parts):
        raise ValueError("FlowCF split contains an empty partition")

    source_users = set(source["user_idx"].astype(int))
    if any(set(part["user_idx"].astype(int)) != source_users for part in parts):
        raise ValueError("Every FlowCF user must occur in train, validation, and test")

    key_columns = ["user_id", "movie_id"]
    source_keys = set(map(tuple, source[key_columns].to_numpy()))
    part_keys = [set(map(tuple, part[key_columns].to_numpy())) for part in parts]
    if len(source_keys) != len(source):
        raise ValueError("Official MovieLens 1M should have unique user-movie pairs")
    if (
        part_keys[0] & part_keys[1]
        or part_keys[0] & part_keys[2]
        or part_keys[1] & part_keys[2]
    ):
        raise ValueError("FlowCF split partitions overlap")
    if set.union(*part_keys) != source_keys:
        raise ValueError("FlowCF split does not preserve every filtered interaction")

    source_counts = source.groupby("user_idx").size()
    actual_counts = [part.groupby("user_idx").size() for part in parts]
    for user_id, total in source_counts.items():
        expected = _ratio_split_counts(int(total))
        actual = tuple(int(counts.loc[user_id]) for counts in actual_counts)
        if actual != expected:
            raise ValueError(
                f"FlowCF split count mismatch for user {user_id}: "
                f"expected {expected}, got {actual}"
            )


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
    if args.protocol == "flowcf" and args.min_positive_rating != 4:
        raise ValueError("FlowCF compatibility requires --min-positive-rating 4")
    if args.protocol == "flowcf" and args.seed != FLOWCF_SEED:
        raise ValueError(f"FlowCF compatibility requires --seed {FLOWCF_SEED}")
    raw_path = args.raw_path or RAW_PATH
    default_output = FLOWCF_OUTPUT_DIR if args.protocol == "flowcf" else OUTPUT_DIR
    output_dir = args.output_dir or default_output
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Download and extract MovieLens 1M so that "
            "ratings.dat is available at this path."
        )
    ratings = load_data(raw_path)
    interactions = filter_positive_interactions(
        ratings, args.min_positive_rating
    )
    if args.protocol == "flowcf":
        interactions = filter_k_core(interactions, min_interactions=5)
        validate_flowcf_dataset(interactions)
    interactions, user2idx, movie2idx = build_id_mapping(interactions)
    if args.protocol == "flowcf":
        train, val, test = flowcf_split(interactions, seed=args.seed)
        validate_flowcf_split(interactions, train, val, test)
    else:
        train, val, test = chronological_split(interactions)
        validate_split(train, val, test)
    save_processed_data(
        train,
        val,
        test,
        user2idx,
        movie2idx,
        output_dir,
        metadata={
            "dataset": "MovieLens 1M",
            "protocol": args.protocol,
            "min_positive_rating": args.min_positive_rating,
            "min_interactions": 5 if args.protocol == "flowcf" else 3,
            "seed": args.seed if args.protocol == "flowcf" else None,
            "split": (
                "random_user_80_10_10"
                if args.protocol == "flowcf"
                else "chronological_leave_two_out"
            ),
            "reference": (
                "https://github.com/chengkai-liu/FlowCF"
                if args.protocol == "flowcf"
                else None
            ),
            "real_item_count": len(movie2idx),
            "recbole_item_count_including_padding": (
                len(movie2idx) + 1 if args.protocol == "flowcf" else None
            ),
        },
    )

    print(
        f"Preprocessing complete | dataset: MovieLens 1M "
        f"| protocol: {args.protocol}"
    )
    print(f"users: {len(user2idx):,} | movies: {len(movie2idx):,}")
    print(f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
