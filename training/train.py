"""Train the two-tower model with in-batch negatives."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import InteractionDataset
from models import TwoTower


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()

    dataset = InteractionDataset(args.processed_dir / "train.csv")
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

    print(
        f"device: {device} | users: {num_users:,} | items: {num_items:,} "
        f"| interactions: {len(dataset):,}"
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch:02d}/{args.epochs:02d} | loss: {loss:.4f}")

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


if __name__ == "__main__":
    main()
