import json
import os
from typing import List, Optional, Union

import tiktoken
import torch

from .base import Tokenizer


class DecoderTokenizer(Tokenizer):
    """Wrapper around a pretrained tiktoken encoding.

    The base encoding's vocabulary is preserved; any user-provided special
    tokens that are not already part of the encoding are appended to the end
    of the vocabulary so that their IDs are deterministic.
    """

    tokenizer_type: str = "decoder"

    def __init__(
        self,
        encoding_name: str = "gpt2",
        pad_token: Optional[str] = "<|pad|>",
        bos_token: Optional[str] = "<|bos|>",
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        base = tiktoken.get_encoding(encoding_name)

        # tiktoken's "gpt2" encoding ships with <|endoftext|>; reuse it for eos
        # unless the caller overrides.
        if eos_token is None:
            eos_token = "<|endoftext|>" if "<|endoftext|>" in base._special_tokens else None

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )
        self.encoding_name = encoding_name

        # extend the encoding with any special tokens not already present
        existing = dict(base._special_tokens)
        extras = {}
        next_id = base.n_vocab
        for tok in (pad_token, bos_token, eos_token, unk_token):
            if tok is None or tok in existing or tok in extras:
                continue
            extras[tok] = next_id
            next_id += 1

        if extras:
            self._enc = tiktoken.Encoding(
                name=f"{encoding_name}_ext",
                pat_str=base._pat_str,
                mergeable_ranks=base._mergeable_ranks,
                special_tokens={**existing, **extras},
            )
        else:
            self._enc = base

        self._allowed_special = set(self._enc._special_tokens.keys())

        self._pad_id = self._lookup(pad_token)
        self._bos_id = self._lookup(bos_token)
        self._eos_id = self._lookup(eos_token)
        self._unk_id = self._lookup(unk_token)

    def _lookup(self, tok: Optional[str]) -> Optional[int]:
        if tok is None:
            return None
        return self._enc.encode(tok, allowed_special={tok})[0]

    # -------------------------
    # vocabulary properties
    # -------------------------
    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_id

    # -------------------------
    # core API
    # -------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self._enc.encode(text, allowed_special=self._allowed_special)
        if add_special_tokens:
            if self._bos_id is not None:
                ids = [self._bos_id] + ids
            if self._eos_id is not None:
                ids = ids + [self._eos_id]
        return ids

    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens:
            special_ids = set(self._enc._special_tokens.values())
            tokens = [t for t in tokens if t not in special_ids]
        return self._enc.decode(tokens)

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg = {
            "tokenizer_type": self.tokenizer_type,
            "encoding_name": self.encoding_name,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DecoderTokenizer":
        with open(os.path.join(path, "tokenizer_config.json"), "r") as f:
            cfg = json.load(f)
        return cls(
            encoding_name=cfg["encoding_name"],
            pad_token=cfg.get("pad_token"),
            bos_token=cfg.get("bos_token"),
            eos_token=cfg.get("eos_token"),
            unk_token=cfg.get("unk_token"),
        )
