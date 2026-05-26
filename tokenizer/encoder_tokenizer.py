import json
import os
from typing import Iterable, List, Optional, Union

import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from .base import Tokenizer


class EncoderTokenizer(Tokenizer):
    """Tokenizer trained from a corpus via the HuggingFace `tokenizers` library.

    Supports BPE (byte-level) and WordPiece subword models. The underlying
    tokenizer is constructed during `train()` or restored via `load()`; calling
    encode/decode before either of those raises a clear error.
    """

    tokenizer_type: str = "encoder"

    def __init__(
        self,
        model_type: str = "bpe",
        pad_token: str = "[PAD]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
    ):
        if model_type not in ("bpe", "wordpiece"):
            raise ValueError(f"model_type must be 'bpe' or 'wordpiece', got {model_type!r}")

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )
        self.model_type = model_type
        self._tokenizer: Optional[HFTokenizer] = None

    # -------------------------
    # internal helpers
    # -------------------------
    def _check_ready(self):
        if self._tokenizer is None:
            raise RuntimeError(
                "EncoderTokenizer is not initialized. Call train(dataset) or load(path) first."
            )

    def _token_to_id(self, tok: Optional[str]) -> Optional[int]:
        if tok is None or self._tokenizer is None:
            return None
        return self._tokenizer.token_to_id(tok)

    # -------------------------
    # vocabulary properties
    # -------------------------
    @property
    def vocab_size(self) -> int:
        self._check_ready()
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._token_to_id(self.pad_token)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._token_to_id(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._token_to_id(self.unk_token)

    # -------------------------
    # core API
    # -------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        self._check_ready()
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_special_tokens:
            bos = self.bos_token_id
            eos = self.eos_token_id
            if bos is not None:
                ids = [bos] + ids
            if eos is not None:
                ids = ids + [eos]
        return ids

    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        self._check_ready()
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # -------------------------
    # training
    # -------------------------
    def train(
        self,
        dataset: Iterable[str],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        **kwargs,
    ) -> None:
        special_tokens = [
            t for t in (self.pad_token, self.unk_token, self.bos_token, self.eos_token) if t
        ]

        if self.model_type == "bpe":
            tok = HFTokenizer(models.BPE(unk_token=self.unk_token))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )
        else:  # wordpiece
            tok = HFTokenizer(models.WordPiece(unk_token=self.unk_token))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
            )

        tok.train_from_iterator(dataset, trainer=trainer)
        self._tokenizer = tok

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str) -> None:
        self._check_ready()
        os.makedirs(path, exist_ok=True)
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))
        cfg = {
            "tokenizer_type": self.tokenizer_type,
            "model_type": self.model_type,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EncoderTokenizer":
        with open(os.path.join(path, "tokenizer_config.json"), "r") as f:
            cfg = json.load(f)
        inst = cls(
            model_type=cfg["model_type"],
            pad_token=cfg["pad_token"],
            bos_token=cfg["bos_token"],
            eos_token=cfg["eos_token"],
            unk_token=cfg["unk_token"],
        )
        inst._tokenizer = HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return inst
