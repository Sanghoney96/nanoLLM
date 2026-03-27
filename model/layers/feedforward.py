import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTFeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_model * 4
        self.linear_1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model, bias=False)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.gelu(x)
        out = self.linear_2(x)

        return out
