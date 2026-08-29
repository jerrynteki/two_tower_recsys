"""Build an exact FAISS inner-product index from trained item embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np


def build(checkpoint: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "item_vectors.npy"
    subprocess.run([sys.executable, "-m", "retrieval.export_embeddings", "--checkpoint", str(checkpoint), "--output", str(vector_path)], check=True)
    vectors = np.load(vector_path).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "items.index"))
    np.save(output_dir / "item_ids.npy", np.arange(len(vectors), dtype=np.int64))
    (output_dir / "metadata.json").write_text(json.dumps({"checkpoint": str(checkpoint), "items": len(vectors), "dimension": vectors.shape[1]}, indent=2))
    vector_path.unlink()
    print(f"indexed {index.ntotal:,} movies -> {(output_dir / 'items.index').resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/two_tower.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/faiss"))
    args = parser.parse_args()
    build(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
