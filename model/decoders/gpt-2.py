import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention import MultiHeadAttention
from model.layers.feedforward import GPTFeedForward
from config.model_config import GPT2Config


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_size, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_size = context_size

        self.layernorm_1 = nn.LayerNorm(self.d_model)
        self.layernorm_2 = nn.LayerNorm(self.d_model)
        self.mha = MultiHeadAttention(
            self.d_model, self.n_heads, self.context_size, dropout, causal=True
        )
        self.ffn = GPTFeedForward(self.d_model, dropout)

    def forward(self, x):
        out = self.layernorm_1(x)
        out = self.mha(out)

        x = out + x

        out = self.layernorm_2(x)
        out = self.ffn(out)

        return out + x


class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.context_size = config.context_size
        self.vocab_size = config.vocab_size
        self.padding_idx = config.padding_idx
        self.num_blocks = config.num_blocks

        self.token_emb = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.padding_idx
        )
        self.pos_emb = nn.Embedding(config.context_size, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    config.d_model, config.n_head, config.context_size, config.dropout
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.layernorm = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight initialization
        self.apply(self._init_weights)

        # weight initialization for out projection of self attention
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("linear_2.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.num_blocks),
                )

        self.out_proj.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        "input shape : (batch_size, context_size)"
        assert x.shape[1] <= self.context_size

        out = self.token_emb(x)
        pos = torch.arange(x.shape[1], device=x.device)
        pos = self.pos_emb(pos)
        out = out + pos

        out = self.dropout(out)
        for block in self.blocks:
            out = block(out)

        out = self.layernorm(out)
        out = self.out_proj(out)
        return out


# load model config
config = GPT2Config()

# 모델 생성
model = GPT2(config)

# 더미 입력 생성
x = torch.randint(0, config.vocab_size, (8, config.context_size))

# forward 테스트
logits = model(x)

print("input shape:", x.shape)
print("output shape:", logits.shape)
