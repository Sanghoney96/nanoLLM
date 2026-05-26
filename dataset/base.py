from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset


class BaseDataset(Dataset, ABC):
    """Abstract base for nanoLLM datasets.

    Subclasses implement `load`, `__getitem__`, and `__len__`.
    `split` is provided generically via torch.utils.data.random_split.
    """

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def __getitem__(self, idx: int): ...

    @abstractmethod
    def __len__(self) -> int: ...

    def split(
        self, train_ratio: float, seed: int = 42
    ) -> Tuple[Subset, Subset]:
        """Random split into (train, val). `train_ratio` is in (0, 1)."""
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

        n = len(self)
        n_train = int(n * train_ratio)
        n_val = n - n_train

        gen = torch.Generator().manual_seed(seed)
        return torch.utils.data.random_split(self, [n_train, n_val], generator=gen)
