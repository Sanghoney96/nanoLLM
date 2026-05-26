from abc import ABC, abstractmethod
from typing import List, Sequence

import torch


class BaseCollator(ABC):
    """Abstract base for batch collators.

    Instances are callable so they can be passed directly as `collate_fn` to
    `torch.utils.data.DataLoader`. Subclasses implement `collate(batch)`.
    """

    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[dict]) -> dict:
        return self.collate(batch)

    @abstractmethod
    def collate(self, batch: List[dict]) -> dict: ...

    # -------------------------
    # shared helpers
    # -------------------------
    @staticmethod
    def pad_sequences(
        sequences: Sequence[torch.Tensor],
        padding_value: int = 0,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """Pad a list of 1D tensors to the same length, returning a 2D tensor."""
        max_len = max(s.size(0) for s in sequences)
        out = torch.full(
            (len(sequences), max_len),
            padding_value,
            dtype=sequences[0].dtype,
        )
        for i, s in enumerate(sequences):
            n = s.size(0)
            if padding_side == "right":
                out[i, :n] = s
            elif padding_side == "left":
                out[i, max_len - n :] = s
            else:
                raise ValueError(f"padding_side must be 'right' or 'left'")
        return out

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.pad_token_id).long()

    def create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Default LM labels: copy of input_ids with pad positions set to ignore_index."""
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = self.ignore_index
        return labels


class LanguageModelingCollator(BaseCollator):
    """Collator for causal LM.

    Accepts batches where each sample has `input_ids` and optionally `labels`.
    If `labels` is missing, it is derived from `input_ids` via `create_labels`.
    Pad positions in `labels` are set to `ignore_index` so cross-entropy skips them.
    """

    def collate(self, batch: List[dict]) -> dict:
        input_ids = self.pad_sequences(
            [b["input_ids"] for b in batch], padding_value=self.pad_token_id
        )
        attention_mask = self.create_attention_mask(input_ids)

        if "labels" in batch[0]:
            labels = self.pad_sequences(
                [b["labels"] for b in batch], padding_value=self.ignore_index
            )
        else:
            labels = self.create_labels(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ClassificationCollator(BaseCollator):
    """Collator for single-text classification.

    Sample schema: `input_ids` (1D), `label` (scalar).
    Output: `input_ids` (B, T), `attention_mask` (B, T), `labels` (B,).
    """

    def collate(self, batch: List[dict]) -> dict:
        input_ids = self.pad_sequences(
            [b["input_ids"] for b in batch], padding_value=self.pad_token_id
        )
        attention_mask = self.create_attention_mask(input_ids)
        labels = torch.stack([b["label"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


