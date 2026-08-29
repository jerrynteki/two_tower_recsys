"""Retrieve recommendations for one raw MovieLens user ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/two_tower.pt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/faiss"))
    args = parser.parse_args()
    with (args.processed_dir / "user2idx.json").open() as handle:
        user_map = {int(k): v for k, v in json.load(handle).items()}
    if args.user_id not in user_map:
        raise ValueError("unknown user ID")
    query_path = args.index_dir / "query.npy"
    subprocess.run([sys.executable, "-m", "retrieval.export_embeddings", "--checkpoint", str(args.checkpoint), "--output", str(query_path), "--user-idx", str(user_map[args.user_id])], check=True)
    query = np.load(query_path).astype("float32")
    query_path.unlink()
    faiss.normalize_L2(query)
    index = faiss.read_index(str(args.index_dir / "items.index"))
    item_ids = np.load(args.index_dir / "item_ids.npy")
    train = pd.read_csv(args.processed_dir / "train.csv")
    seen = set(train.loc[train.user_idx == user_map[args.user_id], "movie_idx"])
    _, positions = index.search(query, min(index.ntotal, args.top_k + len(seen)))
    recommendations = [int(item_ids[p]) for p in positions[0] if int(item_ids[p]) not in seen][: args.top_k]
    movie_map = json.loads((args.processed_dir / "movie2idx.json").read_text())
    raw_by_idx = {idx: raw for raw, idx in movie_map.items()}
    titles = {}
    feature_path = args.processed_dir / "movie_features.csv"
    if feature_path.exists():
        titles = pd.read_csv(feature_path).set_index("movie_idx").title.to_dict()
    for rank, item_idx in enumerate(recommendations, 1):
        print(f"{rank:2d}. movie_id={raw_by_idx[item_idx]} | {titles.get(item_idx, 'title unavailable')}")


if __name__ == "__main__":
    main()
