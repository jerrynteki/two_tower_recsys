"""Repeat one training configuration across seeds and summarize stability."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from evaluation.evaluate import (
    build_seen_items,
    retrieve_baseline_topk,
    single_target_metrics,
)
from training.train import load_catalog_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed-1m")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multi_seed"))
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path("artifacts/multi_seed_trials.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/multi_seed_summary.csv"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--negative-strategy",
        choices=("in_batch", "uniform", "popularity", "hybrid"),
        default="uniform",
    )
    parser.add_argument("--negative-count", type=int, default=64)
    parser.add_argument("--observed-negative-fraction", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="NDCG@10")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def popularity_metrics(
    processed_dir: Path,
    ks: list[int],
    batch_size: int,
) -> dict[str, float]:
    train = pd.read_csv(processed_dir / "train.csv")
    validation = pd.read_csv(processed_dir / "val.csv")
    _, num_items = load_catalog_sizes(processed_dir)
    topk, targets = retrieve_baseline_topk(
        train,
        validation,
        build_seen_items(train),
        num_items,
        max(ks),
        batch_size,
        "popularity",
    )
    return single_target_metrics(topk, targets, ks)


def train_command(args: argparse.Namespace, seed: int, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--processed-dir",
        str(args.processed_dir),
        "--output",
        str(checkpoint),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--embedding-dim",
        str(args.embedding_dim),
        "--learning-rate",
        str(args.learning_rate),
        "--temperature",
        str(args.temperature),
        "--negative-strategy",
        args.negative_strategy,
        "--negative-count",
        str(args.negative_count),
        "--observed-negative-fraction",
        str(args.observed_negative_fraction),
        "--selection-metric",
        args.selection_metric,
        "--ks",
        *[str(k) for k in args.ks],
        "--seed",
        str(seed),
        "--run-name",
        f"{args.negative_strategy}_dim{args.embedding_dim}_seed{seed}",
    ]
    if args.no_tensorboard:
        command.append("--no-tensorboard")
    return command


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = popularity_metrics(args.processed_dir, args.ks, args.batch_size)
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        checkpoint = args.output_dir / f"seed_{seed}.pt"
        print(f"\n=== seed {seed} ===", flush=True)
        subprocess.run(train_command(args, seed, checkpoint), check=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metrics = saved["best_validation_metrics"]
        row: dict[str, object] = {
            "seed": seed,
            "best_epoch": saved["best_epoch"],
            "embedding_dim": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "negative_strategy": args.negative_strategy,
            "negative_count": args.negative_count,
        }
        row.update(metrics)
        for name, value in metrics.items():
            row[f"delta_vs_popularity_{name}"] = value - baseline[name]
        rows.append(row)

    trials = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in trials.columns
        if column.startswith(("Recall@", "HitRate@", "MRR@", "NDCG@", "delta_"))
    ]
    summary = trials[["best_epoch", *metric_columns]].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary = summary.reset_index()

    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.trials_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print("\nPopularity baseline")
    for name, value in baseline.items():
        print(f"{name}: {value:.4f}")
    print("\nThree-seed summary")
    print(summary.to_string(index=False))
    print(f"trials: {args.trials_output.resolve()}")
    print(f"summary: {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
