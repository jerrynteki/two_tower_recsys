"""Train the two-tower model with uniform catalog-negative sampling."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import InteractionDataset
from evaluation.evaluate import build_seen_items, evaluate_retrieval_metrics
from models import TwoTower
from training.checkpointing import (
    capture_random_state,
    restore_random_state,
    save_checkpoint,
)
from training.monitoring import (
    log_configuration,
    log_model_statistics,
    log_validation_metrics,
    timestamped_run_dir,
)
from training.negative_sampling import (
    UniformNegativeSampler,
    train_sampled_epoch,
)


# These settings define the training trajectory and are restored on resume.
RESUME_SETTINGS = (
    "processed_dir",
    "batch_size",
    "embedding_dim",
    "learning_rate",
    "temperature",
    "negative_count",
    "similarity",
    "normalize_embeddings",
    "seed",
    "ks",
    "eval_every",
    "selection_metric",
    "patience",
    "min_delta",
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_catalog_sizes(processed_dir: Path) -> tuple[int, int]:
    with (processed_dir / "user2idx.json").open(encoding="utf-8") as handle:
        num_users = len(json.load(handle))
    with (processed_dir / "movie2idx.json").open(encoding="utf-8") as handle:
        num_items = len(json.load(handle))
    return num_users, num_items


def send_completion_notification(run_name: str, best_epoch: int) -> None:
    """Show a macOS notification after a successful training run.

    ``osascript`` is unavailable on non-macOS platforms, so notification
    failures are intentionally ignored and never affect training results.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                (
                    'display notification "Best checkpoint: epoch '
                    f'{best_epoch}." with title "Two-Tower Training" '
                    f'subtitle "{run_name}" sound name "Glass"'
                ),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed-1m"))
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/two_tower_1m.pt")
    )
    parser.add_argument(
        "--latest-checkpoint", type=Path,
        help="resumable checkpoint path (default: <output stem>.latest<suffix>)",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="resume from a latest checkpoint, restoring its training settings",
    )
    parser.add_argument(
        "--epochs", type=int, default=12,
        help="total epoch limit, including completed epochs",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--negative-count", type=int, default=64)
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
    parser.add_argument("--selection-metric", default="NDCG@10")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/training"))
    parser.add_argument("--run-name", default="two_tower_1m")
    parser.add_argument(
        "--notify",
        action="store_true",
        help="show a macOS notification when training completes successfully",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="train without writing TensorBoard event files",
    )
    return parser.parse_args()


def main() -> None:
    run_started_at = datetime.now().astimezone()
    run_started_clock = time.perf_counter()
    args = parse_args()
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != 1:
            raise ValueError(
                "--resume requires a resumable latest checkpoint, not a best-model export"
            )
        for name in RESUME_SETTINGS:
            setattr(args, name, checkpoint["training_config"][name])
        args.processed_dir = Path(args.processed_dir)
        args.no_early_stopping = (
            args.no_early_stopping or checkpoint["training_config"]["no_early_stopping"]
        )
        if args.epochs < checkpoint["epoch"]:
            raise ValueError("epochs must be at least the checkpoint's completed epoch")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    latest_path = args.latest_checkpoint or args.output.with_name(
        f"{args.output.stem}.latest{args.output.suffix}"
    )
    if args.output.resolve() == latest_path.resolve() or (
        args.resume is not None and args.output.resolve() == args.resume.resolve()
    ):
        raise ValueError("best-model output must differ from resumable checkpoint paths")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if not args.ks or any(k <= 0 for k in args.ks):
        raise ValueError("ks must contain positive integers")
    if args.negative_count <= 0:
        raise ValueError("negative-count must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()

    dataset = InteractionDataset(args.processed_dir / "train.csv")
    train = pd.read_csv(args.processed_dir / "train.csv")
    validation = pd.read_csv(args.processed_dir / "val.csv")
    seen_items = build_seen_items(train)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    num_users, num_items = load_catalog_sizes(args.processed_dir)
    if checkpoint is not None and (
        checkpoint["num_users"] != num_users or checkpoint["num_items"] != num_items
    ):
        raise ValueError("checkpoint catalog sizes do not match the processed data")
    sampler = UniformNegativeSampler(num_items, seen_items, seed=args.seed)
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
        "negative_count": args.negative_count,
        "negative_strategy": "uniform",
        "normalize_embeddings": args.normalize_embeddings,
        "num_items": num_items,
        "num_users": num_users,
        "seed": args.seed,
        "similarity": args.similarity,
        "temperature": args.temperature,
        "selection_metric": args.selection_metric,
        "patience": args.patience,
    }
    if writer:
        log_configuration(writer, config)

    print(
        f"device: {device} | users: {num_users:,} | items: {num_items:,} "
        f"| interactions: {len(dataset):,}"
    )
    final_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metric = float("-inf")
    best_epoch = 0
    evaluations_without_improvement = 0
    start_epoch = 1
    stopped_early = False
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_state_dict = checkpoint["best_model_state_dict"]
        best_metrics = checkpoint["best_validation_metrics"]
        final_metrics = checkpoint["validation_metrics"]
        early_stopping = checkpoint["early_stopping"]
        best_metric = early_stopping["best_metric"]
        best_epoch = early_stopping["best_epoch"]
        evaluations_without_improvement = early_stopping["evaluations_without_improvement"]
        stopped_early = early_stopping["stopped"] and not args.no_early_stopping
        start_epoch = checkpoint["epoch"] + 1
        restore_random_state(checkpoint["random_state"], generator, sampler.generator)
        print(f"resumed: {args.resume.resolve()} | completed epoch: {start_epoch - 1}")
        if stopped_early:
            print("checkpoint already early-stopped; use --no-early-stopping to continue")
    training_config = {name: getattr(args, name) for name in RESUME_SETTINGS}
    training_config["processed_dir"] = str(args.processed_dir.resolve())
    training_config["no_early_stopping"] = args.no_early_stopping
    end_epoch = start_epoch if stopped_early else args.epochs + 1
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        loss = train_sampled_epoch(
            model,
            loader,
            optimizer,
            device,
            sampler,
            args.negative_count,
        )
        train_elapsed = time.perf_counter() - train_started
        validation_elapsed = 0.0
        if writer:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("performance/epoch_seconds", train_elapsed, epoch)
            writer.add_scalar(
                "performance/examples_per_second",
                len(dataset) / train_elapsed,
                epoch,
            )
            log_model_statistics(writer, model, epoch)

        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            validation_started = time.perf_counter()
            final_metrics = evaluate_retrieval_metrics(
                model,
                validation,
                seen_items,
                args.ks,
                args.batch_size,
                device,
            )
            if args.selection_metric not in final_metrics:
                available = ", ".join(final_metrics)
                raise ValueError(
                    f"selection metric {args.selection_metric!r} is unavailable; "
                    f"choose one of: {available}"
                )
            selection_value = final_metrics[args.selection_metric]
            if selection_value > best_metric + args.min_delta:
                best_metric = selection_value
                best_epoch = epoch
                best_metrics = final_metrics.copy()
                best_state_dict = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            recall_text = " | ".join(
                f"{name}: {value:.4f}"
                for name, value in final_metrics.items()
                if name.startswith("Recall")
            )
            if writer:
                log_validation_metrics(writer, final_metrics, epoch)
            validation_elapsed = time.perf_counter() - validation_started
        total_elapsed = time.perf_counter() - epoch_started
        timing_text = f"train: {train_elapsed:.1f}s"
        if should_evaluate:
            timing_text += f" | validation: {validation_elapsed:.1f}s"
        timing_text += f" | total: {total_elapsed:.1f}s"
        message = (
            f"epoch {epoch:02d}/{args.epochs:02d} | {timing_text} "
            f"| loss: {loss:.4f}"
        )
        if should_evaluate:
            message += f" | {recall_text}"
        if writer:
            if should_evaluate:
                writer.add_scalar(
                    "performance/validation_seconds", validation_elapsed, epoch
                )
            writer.add_scalar("performance/total_epoch_seconds", total_elapsed, epoch)
        print(message)
        stopped_early = (
            should_evaluate
            and not args.no_early_stopping
            and evaluations_without_improvement >= args.patience
        )
        save_checkpoint(
            {
                "checkpoint_version": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "num_users": num_users,
                "num_items": num_items,
                "training_config": training_config,
                "best_model_state_dict": best_state_dict,
                "best_validation_metrics": best_metrics,
                "validation_metrics": final_metrics,
                "early_stopping": {
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "stopped": stopped_early,
                },
                "random_state": capture_random_state(generator, sampler.generator),
            },
            latest_path,
        )
        if stopped_early:
            print(
                f"early stopping: {args.selection_metric} did not improve "
                f"for {args.patience} evaluations"
            )
            break

    if best_state_dict is None:
        raise RuntimeError("training completed without a validation result")
    model.load_state_dict(best_state_dict)
    final_metrics = best_metrics

    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "temperature": args.temperature,
            "normalize_embeddings": args.normalize_embeddings,
            "similarity": args.similarity,
            "negative_strategy": "uniform",
            "negative_count": args.negative_count,
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "best_validation_metrics": best_metrics,
            "seed": args.seed,
        },
        args.output,
    )
    if writer:
        writer.add_hparams(config, {f"final/{key}": value for key, value in final_metrics.items()})
        writer.close()
        print(f"tensorboard: {run_dir.resolve()}")
    run_finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - run_started_clock
    elapsed_hours, remainder = divmod(int(elapsed_seconds), 3600)
    elapsed_minutes, elapsed_seconds_remainder = divmod(remainder, 60)
    print(
        f"best epoch: {best_epoch} | {args.selection_metric}: {best_metric:.4f}"
        f" | started: {run_started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f" | finished: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f" | duration: {elapsed_hours}h {elapsed_minutes}m "
        f"{elapsed_seconds_remainder}s"
    )
    print(f"checkpoint: {args.output.resolve()}")
    saved_latest_path = args.resume if start_epoch == end_epoch else latest_path
    print(f"resumable checkpoint: {saved_latest_path.resolve()}")
    if args.notify:
        send_completion_notification(args.run_name, best_epoch)


if __name__ == "__main__":
    main()
