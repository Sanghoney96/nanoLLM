"""
Pipeline integration tests: raw text → Dataset → Collator → DataLoader

Covers:
  - LanguageModelingDataset  + LanguageModelingCollator
  - TextClassificationDataset + ClassificationCollator
"""

import json
import pytest
import torch
from torch.utils.data import DataLoader

from tokenizer.decoder_tokenizer import DecoderTokenizer
from dataset.language_modeling import LanguageModelingDataset
from dataset.text_classification import TextClassificationDataset
from dataset.collators import LanguageModelingCollator, ClassificationCollator


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    return DecoderTokenizer(encoding_name="gpt2")


# ---------------------------------------------------------------------------
# LanguageModelingDataset
# ---------------------------------------------------------------------------

class TestLanguageModelingDataset:
    CONTEXT = 8

    @pytest.fixture
    def txt_file(self, tmp_path):
        text = "the quick brown fox jumps over the lazy dog " * 20
        p = tmp_path / "corpus.txt"
        p.write_text(text, encoding="utf-8")
        return str(p)

    @pytest.fixture
    def jsonl_file(self, tmp_path):
        docs = [
            {"text": "hello world " * 10},
            {"text": "the quick brown fox " * 10},
            {"text": "natural language processing " * 10},
        ]
        p = tmp_path / "corpus.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
        return str(p)

    @pytest.fixture
    def lm_dataset(self, tokenizer, txt_file):
        return LanguageModelingDataset(tokenizer, context_size=self.CONTEXT, path=txt_file)

    def test_length(self, lm_dataset, tokenizer, txt_file):
        text = open(txt_file).read()
        total_tokens = len(tokenizer.encode(text))
        assert len(lm_dataset) == total_tokens - self.CONTEXT

    def test_sample_shapes(self, lm_dataset):
        sample = lm_dataset[0]
        assert sample["input_ids"].shape == (self.CONTEXT,)
        assert sample["labels"].shape == (self.CONTEXT,)

    def test_label_is_shifted(self, lm_dataset):
        # labels[i] must equal input_ids[i+1] for all but the last position
        sample = lm_dataset[0]
        assert torch.equal(sample["labels"][:-1], sample["input_ids"][1:])

    def test_consecutive_samples_overlap(self, lm_dataset):
        # sample[i+1].input_ids[0] == sample[i].input_ids[1]
        s0 = lm_dataset[0]["input_ids"]
        s1 = lm_dataset[1]["input_ids"]
        assert torch.equal(s0[1:], s1[:-1])

    def test_loads_jsonl(self, tokenizer, jsonl_file):
        ds = LanguageModelingDataset(tokenizer, context_size=self.CONTEXT, path=jsonl_file)
        assert len(ds) > 0
        sample = ds[0]
        assert "input_ids" in sample and "labels" in sample

    def test_unsupported_format_raises(self, tokenizer, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            LanguageModelingDataset(tokenizer, context_size=4, path=str(p))


class TestLanguageModelingCollator:
    CONTEXT = 8
    PAD = 0

    @pytest.fixture
    def collator(self, tokenizer):
        return LanguageModelingCollator(pad_token_id=tokenizer.pad_token_id)

    @pytest.fixture
    def batch(self, tokenizer):
        # 직접 다른 길이의 샘플을 구성해 collator만 단독 테스트
        return [
            {"input_ids": torch.arange(6), "labels": torch.arange(1, 7)},
            {"input_ids": torch.arange(4), "labels": torch.arange(1, 5)},
            {"input_ids": torch.arange(8), "labels": torch.arange(1, 9)},
        ]

    def test_output_keys(self, collator, batch):
        out = collator(batch)
        assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_output_shapes(self, collator, batch):
        out = collator(batch)
        B, T = 3, 8  # longest sample has length 8
        assert out["input_ids"].shape == (B, T)
        assert out["attention_mask"].shape == (B, T)
        assert out["labels"].shape == (B, T)

    def test_pad_positions_masked(self, collator, batch):
        out = collator(batch)
        # 두 번째 샘플은 길이 4 → 뒤 4칸이 pad
        assert out["attention_mask"][1, 4:].sum() == 0
        assert out["attention_mask"][1, :4].sum() == 4

    def test_pad_positions_in_labels_are_ignore_index(self, collator, batch):
        out = collator(batch)
        assert (out["labels"][1, 4:] == -100).all()


class TestLanguageModelingDataLoader:
    CONTEXT = 8
    BATCH = 4

    @pytest.fixture
    def loader(self, tokenizer, tmp_path):
        text = "the quick brown fox jumps over the lazy dog " * 50
        p = tmp_path / "corpus.txt"
        p.write_text(text)
        ds = LanguageModelingDataset(tokenizer, context_size=self.CONTEXT, path=str(p))
        collator = LanguageModelingCollator(pad_token_id=tokenizer.pad_token_id)
        return DataLoader(ds, batch_size=self.BATCH, collate_fn=collator, drop_last=True)

    def test_batch_shapes(self, loader):
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (self.BATCH, self.CONTEXT)
        assert batch["labels"].shape == (self.BATCH, self.CONTEXT)
        assert batch["attention_mask"].shape == (self.BATCH, self.CONTEXT)

    def test_all_batches_iterable(self, loader):
        for batch in loader:
            assert "input_ids" in batch


# ---------------------------------------------------------------------------
# TextClassificationDataset
# ---------------------------------------------------------------------------

CLASSIFY_SAMPLES = [
    {"text": "I love this product", "label": "positive"},
    {"text": "This is terrible", "label": "negative"},
    {"text": "It is okay", "label": "neutral"},
    {"text": "Absolutely amazing", "label": "positive"},
    {"text": "Would not recommend", "label": "negative"},
]


class TestTextClassificationDataset:
    MAX_LEN = 16

    @pytest.fixture
    def jsonl_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(s) for s in CLASSIFY_SAMPLES))
        return str(p)

    @pytest.fixture
    def cls_dataset(self, tokenizer, jsonl_file):
        return TextClassificationDataset(tokenizer, max_length=self.MAX_LEN, path=jsonl_file)

    def test_length(self, cls_dataset):
        assert len(cls_dataset) == len(CLASSIFY_SAMPLES)

    def test_sample_shapes(self, cls_dataset):
        sample = cls_dataset[0]
        assert sample["input_ids"].shape == (self.MAX_LEN,)
        assert sample["attention_mask"].shape == (self.MAX_LEN,)
        assert sample["label"].shape == ()

    def test_attention_mask_matches_real_tokens(self, cls_dataset, tokenizer):
        sample = cls_dataset[0]
        real_len = sample["attention_mask"].sum().item()
        encoded_len = min(len(tokenizer.encode(CLASSIFY_SAMPLES[0]["text"])), self.MAX_LEN)
        assert real_len == encoded_len

    def test_pad_tokens_at_end(self, cls_dataset, tokenizer):
        sample = cls_dataset[0]
        pad_id = tokenizer.pad_token_id
        real_len = sample["attention_mask"].sum().item()
        assert (sample["input_ids"][real_len:] == pad_id).all()

    def test_string_labels_mapped_to_int(self, cls_dataset):
        # 3개의 고유 레이블이 0,1,2로 매핑되어야 함
        assert cls_dataset.num_labels == 3
        labels = {cls_dataset[i]["label"].item() for i in range(len(cls_dataset))}
        assert labels == {0, 1, 2}

    def test_integer_labels_preserved(self, tokenizer, tmp_path):
        samples = [{"text": "hello", "label": 5}, {"text": "world", "label": 3}]
        p = tmp_path / "int_labels.jsonl"
        p.write_text("\n".join(json.dumps(s) for s in samples))
        ds = TextClassificationDataset(tokenizer, max_length=8, path=str(p))
        assert ds[0]["label"].item() == 5
        assert ds[1]["label"].item() == 3

    def test_non_jsonl_raises(self, tokenizer, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Only .jsonl"):
            TextClassificationDataset(tokenizer, max_length=8, path=str(p))


class TestClassificationCollator:
    MAX_LEN = 16

    @pytest.fixture
    def collator(self, tokenizer):
        return ClassificationCollator(pad_token_id=tokenizer.pad_token_id)

    @pytest.fixture
    def cls_dataset(self, tokenizer, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(s) for s in CLASSIFY_SAMPLES))
        return TextClassificationDataset(tokenizer, max_length=self.MAX_LEN, path=str(p))

    def test_output_keys(self, collator, cls_dataset):
        batch = [cls_dataset[i] for i in range(3)]
        out = collator(batch)
        assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_batch_shapes(self, collator, cls_dataset):
        batch = [cls_dataset[i] for i in range(3)]
        out = collator(batch)
        assert out["input_ids"].shape == (3, self.MAX_LEN)
        assert out["attention_mask"].shape == (3, self.MAX_LEN)
        assert out["labels"].shape == (3,)

    def test_labels_are_scalar_per_sample(self, collator, cls_dataset):
        batch = [cls_dataset[i] for i in range(len(cls_dataset))]
        out = collator(batch)
        assert out["labels"].dtype == torch.long


class TestClassificationDataLoader:
    MAX_LEN = 16
    BATCH = 3

    @pytest.fixture
    def loader(self, tokenizer, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(s) for s in CLASSIFY_SAMPLES))
        ds = TextClassificationDataset(tokenizer, max_length=self.MAX_LEN, path=str(p))
        collator = ClassificationCollator(pad_token_id=tokenizer.pad_token_id)
        return DataLoader(ds, batch_size=self.BATCH, collate_fn=collator)

    def test_batch_shapes(self, loader):
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (self.BATCH, self.MAX_LEN)
        assert batch["labels"].shape == (self.BATCH,)

    def test_all_batches_iterable(self, loader):
        for batch in loader:
            assert "input_ids" in batch


# ---------------------------------------------------------------------------
# split() integration
# ---------------------------------------------------------------------------

class TestDatasetSplit:
    def test_lm_split_sizes(self, tokenizer, tmp_path):
        text = "the quick brown fox jumps over the lazy dog " * 50
        p = tmp_path / "corpus.txt"
        p.write_text(text)
        ds = LanguageModelingDataset(tokenizer, context_size=8, path=str(p))
        train, val = ds.split(train_ratio=0.8)
        assert len(train) + len(val) == len(ds)
        assert abs(len(train) / len(ds) - 0.8) < 0.05

    def test_cls_split_no_overlap(self, tokenizer, tmp_path):
        samples = [{"text": f"sample {i}", "label": i % 3} for i in range(30)]
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(s) for s in samples))
        ds = TextClassificationDataset(tokenizer, max_length=16, path=str(p))
        train, val = ds.split(train_ratio=0.8)
        train_idx = set(train.indices)
        val_idx = set(val.indices)
        assert train_idx.isdisjoint(val_idx)
