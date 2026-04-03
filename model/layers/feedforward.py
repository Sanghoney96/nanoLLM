import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_model * 4
        self.linear_1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        out = self.dropout(x)

        return out
