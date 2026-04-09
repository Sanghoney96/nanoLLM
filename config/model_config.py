from dataclasses import dataclass, asdict
import json
import os
from typing import Type, Dict


CONFIG_REGISTRY: Dict[str, Type["ModelConfig"]] = {}


@dataclass
class ModelConfig:
    model_type: str = "base"

    # -------------------------
    # auto registry
    # -------------------------
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_type"):
            CONFIG_REGISTRY[cls.model_type] = cls

    # -------------------------
    # dict
    # -------------------------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    # -------------------------
    # save / load (HF-style)
    # -------------------------
    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str):
        path = os.path.join(load_dir, "config.json")

        with open(path, "r") as f:
            config_dict = json.load(f)

        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("config.json must contain 'model_type'")

        config_class = CONFIG_REGISTRY.get(model_type)
        if config_class is None:
            raise ValueError(f"Unknown model_type: {model_type}")

        return config_class.from_dict(config_dict)

    # -------------------------
    # validation hook
    # -------------------------
    def __post_init__(self):
        pass


@dataclass
class GPT2Config(ModelConfig):
    model_type: str = "gpt2"

    n_head: int = 12
    num_blocks: int = 12
    d_model: int = 768
    context_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.1
    padding_idx: int = 1

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
