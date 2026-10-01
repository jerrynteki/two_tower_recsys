"""Compare representation and scoring choices under a fixed training setup."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import InteractionDataset
from evaluation.evaluate import (
    build_seen_items,
    evaluate_retrieval_metrics,
    select_device,
)
from models import TwoTower
from training.negative_sampling import UniformNegativeSampler, train_sampled_epoch
from training.train import load_catalog_sizes
from training.monitoring import (
    log_configuration,
    log_model_statistics,
    log_validation_metrics,
    timestamped_run_dir,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    embedding_dim: int = 64
    temperature: float = 0.07
    normalize_embeddings: bool = True
    similarity: str = "dot"


EXPERIMENTS = (
    Experiment("baseline"),
    Experiment("dim_32", embedding_dim=32),
    Experiment("dim_128", embedding_dim=128),
    Experiment("temperature_003", temperature=0.03),
    Experiment("temperature_020", temperature=0.20),
    Experiment("raw_dot", normalize_embeddings=False),
    Experiment("raw_cosine", normalize_embeddings=False, similarity="cosine"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed-1m")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model_experiments_1m.csv")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-count", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("runs/model_experiments_1m")
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ks or any(k <= 0 for k in args.ks):
        raise ValueError("ks must contain positive integers")
    device = select_device()
    dataset = InteractionDataset(args.processed_dir / "train.csv")
    train = pd.read_csv(args.processed_dir / "train.csv")
    validation = pd.read_csv(args.processed_dir / "val.csv")
    seen_items = build_seen_items(train)
    num_users, num_items = load_catalog_sizes(args.processed_dir)
    records: list[dict[str, object]] = []

    print(f"device: {device} | experiments: {len(EXPERIMENTS)}")
    session_dir = timestamped_run_dir(args.log_dir, "comparison")
    for experiment in EXPERIMENTS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model = TwoTower(
            num_users,
            num_items,
            embedding_dim=experiment.embedding_dim,
            temperature=experiment.temperature,
            normalize_embeddings=experiment.normalize_embeddings,
            similarity=experiment.similarity,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        sampler = UniformNegativeSampler(num_items, seen_items, seed=args.seed)
        writer = None
        config = {
            **asdict(experiment),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "negative_count": args.negative_count,
            "seed": args.seed,
        }
        if not args.no_tensorboard:
            writer = SummaryWriter(log_dir=session_dir / experiment.name)
            log_configuration(writer, config)

        loss = float("nan")
        metrics: dict[str, float] = {}
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            loss = train_sampled_epoch(
                model,
                loader,
                optimizer,
                device,
                sampler,
                args.negative_count,
            )
            elapsed = time.perf_counter() - started
            metrics = evaluate_retrieval_metrics(
                model,
                validation,
                seen_items,
                args.ks,
                args.batch_size,
                device,
            )
            if writer:
                writer.add_scalar("train/loss", loss, epoch)
                writer.add_scalar("performance/epoch_seconds", elapsed, epoch)
                writer.add_scalar("performance/examples_per_second", len(dataset) / elapsed, epoch)
                log_model_statistics(writer, model, epoch)
                log_validation_metrics(writer, metrics, epoch)
        record: dict[str, object] = asdict(experiment)
        record.update({"epochs": args.epochs, "training_loss": loss, **metrics})
        records.append(record)
        metric_text = " | ".join(
            f"{name}: {value:.4f}"
            for name, value in metrics.items()
            if name.startswith("Recall")
        )
        print(f"{experiment.name:>16} | loss: {loss:.4f} | {metric_text}")
        if writer:
            writer.add_hparams(config, {f"final/{key}": value for key, value in metrics.items()})
            writer.close()

    results = pd.DataFrame(records).sort_values("Recall@10", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"results: {args.output.resolve()}")
    if not args.no_tensorboard:
        print(f"tensorboard: {session_dir.resolve()}")


if __name__ == "__main__":
    main()
