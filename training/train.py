"""Train the two-tower model with in-batch negatives."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import InteractionDataset
from evaluation.evaluate import build_seen_items, retrieve_topk, single_target_metrics
from models import TwoTower
from training.monitoring import (
    log_configuration,
    log_model_statistics,
    log_validation_metrics,
    timestamped_run_dir,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: TwoTower,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    for user_ids, item_ids in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels = torch.arange(len(user_ids), device=device)

        optimizer.zero_grad()
        logits = model.in_batch_logits(user_ids, item_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(user_ids)
        total_examples += len(user_ids)

    return total_loss / total_examples


def load_catalog_sizes(processed_dir: Path) -> tuple[int, int]:
    with (processed_dir / "user2idx.json").open(encoding="utf-8") as handle:
        num_users = len(json.load(handle))
    with (processed_dir / "movie2idx.json").open(encoding="utf-8") as handle:
        num_items = len(json.load(handle))
    return num_users, num_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/two_tower.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--similarity", choices=("dot", "cosine"), default="dot")
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize_embeddings",
        help="disable L2 normalization at each tower output",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/training"))
    parser.add_argument("--run-name", default="two_tower")
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="train without writing TensorBoard event files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if not args.ks or any(k <= 0 for k in args.ks):
        raise ValueError("ks must contain positive integers")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()

    dataset = InteractionDataset(args.processed_dir / "train.csv")
    train = pd.read_csv(args.processed_dir / "train.csv")
    validation = pd.read_csv(args.processed_dir / "val.csv")
    seen_items = build_seen_items(train)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    num_users, num_items = load_catalog_sizes(args.processed_dir)
    model = TwoTower(
        num_users,
        num_items,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        normalize_embeddings=args.normalize_embeddings,
        similarity=args.similarity,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_dir = timestamped_run_dir(args.log_dir, args.run_name)
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=run_dir)
    config = {
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "normalize_embeddings": args.normalize_embeddings,
        "num_items": num_items,
        "num_users": num_users,
        "seed": args.seed,
        "similarity": args.similarity,
        "temperature": args.temperature,
    }
    if writer:
        log_configuration(writer, config)

    print(
        f"device: {device} | users: {num_users:,} | items: {num_items:,} "
        f"| interactions: {len(dataset):,}"
    )
    final_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        loss = train_one_epoch(model, loader, optimizer, device)
        elapsed = time.perf_counter() - started
        message = f"epoch {epoch:02d}/{args.epochs:02d} | loss: {loss:.4f}"
        if writer:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("performance/epoch_seconds", elapsed, epoch)
            writer.add_scalar("performance/examples_per_second", len(dataset) / elapsed, epoch)
            log_model_statistics(writer, model, epoch)

        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            topk_items, targets = retrieve_topk(
                model,
                validation,
                seen_items,
                max(args.ks),
                args.batch_size,
                device,
            )
            final_metrics = single_target_metrics(topk_items, targets, args.ks)
            recall_text = " | ".join(
                f"{name}: {value:.4f}"
                for name, value in final_metrics.items()
                if name.startswith("Recall")
            )
            message = f"{message} | {recall_text}"
            if writer:
                log_validation_metrics(writer, final_metrics, epoch)
        print(message)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "temperature": args.temperature,
            "normalize_embeddings": args.normalize_embeddings,
            "similarity": args.similarity,
        },
        args.output,
    )
    print(f"checkpoint: {args.output.resolve()}")
    if writer:
        writer.add_hparams(config, {f"final/{key}": value for key, value in final_metrics.items()})
        writer.close()
        print(f"tensorboard: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
