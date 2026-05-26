from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union

import torch


class Tokenizer(ABC):
    tokenizer_type: str = "base"

    def __init__(
        self,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

    # -------------------------
    # vocabulary properties
    # -------------------------
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def pad_token_id(self) -> Optional[int]: ...

    @property
    @abstractmethod
    def bos_token_id(self) -> Optional[int]: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> Optional[int]: ...

    @property
    @abstractmethod
    def unk_token_id(self) -> Optional[int]: ...

    # -------------------------
    # core API
    # -------------------------
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]: ...

    @abstractmethod
    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str: ...

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        padding: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding_side: str = "right",
        return_tensors: bool = True,
    ) -> dict:
        """Encode a batch of texts.

        Returns dict with keys:
          - input_ids: 2D tensor or list[list[int]]
          - attention_mask: same shape, 1 for real tokens, 0 for padding
        """
        all_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        if truncation and max_length is not None:
            all_ids = [ids[:max_length] for ids in all_ids]

        masks = [[1] * len(ids) for ids in all_ids]

        if padding:
            pad_id = self.pad_token_id
            if pad_id is None:
                raise ValueError(
                    "padding=True but pad_token_id is not set on this tokenizer."
                )

            target_len = max(len(ids) for ids in all_ids) if all_ids else 0
            if max_length is not None:
                target_len = min(target_len, max_length)

            padded_ids, padded_masks = [], []
            for ids, mask in zip(all_ids, masks):
                pad_n = target_len - len(ids)
                if padding_side == "right":
                    padded_ids.append(ids + [pad_id] * pad_n)
                    padded_masks.append(mask + [0] * pad_n)
                elif padding_side == "left":
                    padded_ids.append([pad_id] * pad_n + ids)
                    padded_masks.append([0] * pad_n + mask)
                else:
                    raise ValueError(f"padding_side must be 'right' or 'left'")
            all_ids, masks = padded_ids, padded_masks

        if return_tensors:
            if not padding:
                raise ValueError("return_tensors=True requires padding=True")
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": all_ids, "attention_mask": masks}

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]

    # -------------------------
    # training / persistence
    # -------------------------
    def train(self, dataset: Iterable[str], **kwargs) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training."
        )

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Tokenizer": ...
