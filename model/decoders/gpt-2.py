import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention import MultiHeadAttention
from model.layers.feedforward import GPTFeedForward


class GPTBlock(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.layernorm_1 = nn.LayerNorm(self.d_model)
        self.layernorm_2 = nn.LayerNorm(self.d_model)
        self.mha = MultiHeadAttention(self.d_model, self.n_heads, dropout, causal=True)
        self.ffn = GPTFeedForward(self.d_model, dropout)

    def forward(self, x):
        out = self.layernorm_1(x)
        out = self.mha(out)

        x = out + x

        out = self.layernorm_2(x)
        out = self.ffn(out)

        return out + x


class GPT2(nn.Module):

    def __init__(
        self,
        n_head,
        d_model,
        context_size,
        vocab_size,
        num_blocks,
        dropout=0.0,
        padding_idx=1,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.token_emb = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=self.padding_idx
        )
        self.pos_emb = nn.Embedding(self.context_size, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [GPTBlock(self.n_head, self.d_model, dropout) for _ in range(num_blocks)]
        )

        self.layernorm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.out_proj.weight = self.token_emb.weight

    def forward(self, x):
        "input shape : (batch_size, context_size)"
        assert self.context_size == x.shape[1]

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


# 하이퍼파라미터 (작게 설정)
batch_size = 4
context_size = 128
vocab_size = 100
d_model = 64
n_head = 4
num_blocks = 2

# 모델 생성
model = GPT2(
    n_head=n_head,
    d_model=d_model,
    context_size=context_size,
    vocab_size=vocab_size,
    num_blocks=num_blocks,
    dropout=0.1,
)

# 더미 입력 생성
x = torch.randint(0, vocab_size, (batch_size, context_size))

# forward 테스트
logits = model(x)

print("input shape:", x.shape)
print("output shape:", logits.shape)
