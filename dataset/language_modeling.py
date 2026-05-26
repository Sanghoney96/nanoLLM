import json
from pathlib import Path
from typing import Optional

import torch

from .base import BaseDataset


class LanguageModelingDataset(BaseDataset):
    """Causal language modeling dataset (packed stream).

    The full corpus is tokenized once and concatenated into a single 1D buffer
    of token ids. Each sample is a contiguous slice of length `context_size + 1`,
    yielding (input_ids, labels) where labels are input_ids shifted by one.

    Documents in jsonl files are joined with the tokenizer's eos token (if
    available) so that the model sees explicit document boundaries.
    """

    def __init__(
        self,
        tokenizer,
        context_size: int,
        path: Optional[str] = None,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.text_key = text_key
        self.ids: torch.Tensor = torch.empty(0, dtype=torch.long)

        if path is not None:
            self.load(path)

    def load(self, path: str) -> None:
        p = Path(path)
        if p.suffix == ".txt":
            with open(p, "r", encoding="utf-8") as f:
                docs = [f.read()]
        elif p.suffix == ".jsonl":
            docs = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    docs.append(json.loads(line)[self.text_key])
        else:
            raise ValueError(
                f"Unsupported file format: {p.suffix!r}. Use .txt or .jsonl"
            )

        eos_id = self.tokenizer.eos_token_id
        all_ids = []
        for i, doc in enumerate(docs):
            all_ids.extend(self.tokenizer.encode(doc))
            if eos_id is not None and i < len(docs) - 1:
                all_ids.append(eos_id)

        self.ids = torch.tensor(all_ids, dtype=torch.long)

    def __len__(self) -> int:
        # need context_size + 1 tokens (input + shifted target)
        return max(0, self.ids.numel() - self.context_size)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.ids[idx : idx + self.context_size + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
