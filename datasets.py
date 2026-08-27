"""Datasets for positive user-item interactions."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return one indexed user and their observed positive item."""

    def __init__(self, csv_path: str | Path) -> None:
        interactions = pd.read_csv(csv_path, usecols=["user_idx", "movie_idx"])
        self.user_ids = torch.as_tensor(
            interactions["user_idx"].to_numpy(), dtype=torch.long
        )
        self.item_ids = torch.as_tensor(
            interactions["movie_idx"].to_numpy(), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_ids[index], self.item_ids[index]

