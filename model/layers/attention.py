import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, block_size=1024):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.block_size = block_size

        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                    1, 1, block_size, block_size
                ),
            )

        self.W_qkv = nn.Linear(self.d_model, self.d_model * 3, bias=True)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (batch_size, input_len, d_model) → (batch_size, heads, input_len, head_dim)
        """
        batch_size, input_len = x.shape[0], x.shape[1]

        x = x.view(batch_size, input_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def concat_heads(self, x):
        """
        (batch_size, n_heads, input_len, head_dim) → (batch_size, input_len, d_model)
        """
        batch_size, n_heads, input_len, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, input_len, n_heads * head_dim)

        return x

    def generate_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask

    def scaled_dot_product_attention(self, q, k, v, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        return out

    def forward(self, x, padding_mask=None):
        """
        x: (batch_size, input_len, d_model)
        """

        input_len = x.shape[1]

        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        mask = None

        # causal mask
        if self.causal:
            mask = self.causal_mask[:, :, :input_len, :input_len]

        # padding mask
        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :].bool()
            if mask is None:
                mask = padding_mask
            else:
                mask = mask & padding_mask

        heads = self.scaled_dot_product_attention(q, k, v, mask)

        concat = self.concat_heads(heads)
        out = self.W_o(concat)
        out = self.res_dropout(out)

        return out
