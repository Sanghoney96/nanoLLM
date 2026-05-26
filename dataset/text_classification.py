import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from .base import BaseDataset


class TextClassificationDataset(BaseDataset):
    """Single-text classification dataset.

    Source format: jsonl, one sample per line with `text` and `label` fields
    (keys configurable). String labels are auto-mapped to ints in sorted order;
    integer labels are kept as-is.

    `__getitem__` returns a dict with fixed-length `input_ids`, `attention_mask`,
    and `label` tensors so the default DataLoader collator works without a
    custom collate_fn.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        path: Optional[str] = None,
        text_key: str = "text",
        label_key: str = "label",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.label_key = label_key
        self.samples: List[Tuple[str, Union[int, str]]] = []
        self.label_to_id: Dict[str, int] = {}

        if path is not None:
            self.load(path)

    def load(self, path: str) -> None:
        p = Path(path)
        if p.suffix != ".jsonl":
            raise ValueError(f"Only .jsonl supported, got {p.suffix!r}")

        labels_seen = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text, label = obj[self.text_key], obj[self.label_key]
                self.samples.append((text, label))
                if label not in labels_seen:
                    labels_seen.append(label)

        if labels_seen and isinstance(labels_seen[0], str):
            self.label_to_id = {l: i for i, l in enumerate(sorted(labels_seen))}

    @property
    def num_labels(self) -> int:
        if self.label_to_id:
            return len(self.label_to_id)
        # int labels: assume contiguous 0..max
        if self.samples:
            return max(int(s[1]) for s in self.samples) + 1
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]

        ids = self.tokenizer.encode(text)[: self.max_length]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id for classification datasets."
            )

        n_pad = self.max_length - len(ids)
        attention = [1] * len(ids) + [0] * n_pad
        ids = ids + [pad_id] * n_pad

        if self.label_to_id:
            label = self.label_to_id[label]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "label": torch.tensor(int(label), dtype=torch.long),
        }
