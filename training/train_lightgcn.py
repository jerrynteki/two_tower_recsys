"""Train LightGCN with BPR on the MovieLens implicit interaction graph."""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from evaluation.evaluate import build_seen_items, mask_seen_items, multi_target_metrics
from models.lightgcn import LightGCN, normalized_bipartite_adjacency
from training.train import load_catalog_sizes


def select_lightgcn_device() -> torch.device:
    """Choose an accelerator only when sparse graph propagation is supported."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            adjacency = torch.sparse_coo_tensor(
                torch.tensor([[0], [1]], device="mps"),
                torch.ones(1, device="mps"),
                (2, 2),
            ).coalesce()
            torch.sparse.mm(adjacency, torch.ones((2, 1), device="mps"))
            return torch.device("mps")
        except (RuntimeError, NotImplementedError):
            print("MPS sparse graph propagation is unavailable; using CPU")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed-1m"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/lightgcn_ml1m.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/training"))
    parser.add_argument("--run-name", default="lightgcn_ml1m_d64_l3_s42")
    return parser.parse_args()


def sample_negatives(
    user_ids: torch.Tensor,
    seen_mask: torch.Tensor,
    num_items: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one uniform unseen item per user, using vectorized rejection."""
    negatives = torch.randint(num_items, (len(user_ids),), generator=generator)
    invalid = seen_mask[user_ids, negatives]
    while invalid.any():
        negatives[invalid] = torch.randint(
            num_items, (int(invalid.sum()),), generator=generator
        )
        invalid = seen_mask[user_ids, negatives]
    return negatives


@torch.no_grad()
def evaluate(
    model: LightGCN,
    data: pd.DataFrame,
    seen_items: dict[int, set[int]],
    ks: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    user_embeddings, item_embeddings = model.propagated_embeddings()
    relevant_items = build_seen_items(data)
    user_ids = torch.tensor(sorted(relevant_items), dtype=torch.long)
    topk_parts = []
    max_k = max(ks)
    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start : start + batch_size].to(device)
        scores = user_embeddings[batch_users] @ item_embeddings.T
        scores = mask_seen_items(scores, batch_users, seen_items)
        topk_parts.append(scores.topk(max_k, dim=1).indices.cpu())
    return multi_target_metrics(torch.cat(topk_parts), user_ids, relevant_items, ks)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.embedding_dim <= 0 or args.layers < 0:
        raise ValueError("embedding-dim must be positive and layers nonnegative")
    if args.learning_rate <= 0 or args.regularization < 0:
        raise ValueError("learning-rate must be positive and regularization nonnegative")

    started_at = datetime.now().astimezone()
    clock_started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_lightgcn_device()

    train = pd.read_csv(args.processed_dir / "train.csv")
    validation = pd.read_csv(args.processed_dir / "val.csv")
    num_users, num_items = load_catalog_sizes(args.processed_dir)
    train_users = torch.tensor(train["user_idx"].to_numpy(), dtype=torch.long)
    train_items = torch.tensor(train["movie_idx"].to_numpy(), dtype=torch.long)
    adjacency = normalized_bipartite_adjacency(
        num_users, num_items, train_users, train_items
    )
    model = LightGCN(
        num_users, num_items, adjacency,
        embedding_dim=args.embedding_dim, num_layers=args.layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(train_users, train_items),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    seen_mask = torch.zeros((num_users, num_items), dtype=torch.bool)
    seen_mask[train_users, train_items] = True
    negative_generator = torch.Generator().manual_seed(args.seed)
    seen_items = build_seen_items(train)
    writer = SummaryWriter(log_dir=args.log_dir / args.run_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device: {device} | users: {num_users:,} | items: {num_items:,} "
        f"| train edges: {len(train):,} | LightGCN layers: {args.layers}"
    )
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        examples = 0
        for users, positives in loader:
            negatives = sample_negatives(users, seen_mask, num_items, negative_generator)
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.bpr_loss(users, positives, negatives, args.regularization)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(users)
            examples += len(users)
        train_loss = total_loss / examples
        metrics = evaluate(
            model, validation, seen_items, (10, 20), args.eval_batch_size, device
        )
        score = metrics["NDCG@10"]
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed = time.perf_counter() - epoch_start
        writer.add_scalar("train/loss", train_loss, epoch)
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        writer.add_scalar("performance/epoch_seconds", elapsed, epoch)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | {elapsed:.1f}s "
            f"| loss: {train_loss:.4f} | "
            + " | ".join(f"{name}: {value:.4f}" for name, value in metrics.items()),
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"early stopping: NDCG@10 did not improve for {args.patience} epochs")
            break

    writer.close()
    if best_state is None:
        raise RuntimeError("training completed without a validation result")
    torch.save(
        {
            "model_state_dict": best_state,
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.layers,
            "regularization": args.regularization,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "seed": args.seed,
            "architecture": "lightgcn",
        },
        args.output,
    )
    finished_at = datetime.now().astimezone()
    elapsed = int(time.perf_counter() - clock_started)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"best epoch: {best_epoch} | NDCG@10: {best_score:.4f} "
        f"| started: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"| finished: {finished_at.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"| duration: {hours}h {minutes}m {seconds}s"
    )
    print(f"checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
