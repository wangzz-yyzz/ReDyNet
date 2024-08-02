import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, qkv_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.qkv_dim = qkv_dim

        self.q = nn.Linear(in_dim, qkv_dim, bias=False)
        self.k = nn.Linear(in_dim, qkv_dim, bias=False)
        self.v = nn.Linear(in_dim, qkv_dim, bias=False)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.qkv_dim).float())
        scores = torch.nn.functional.softmax(scores, dim=-1)
        x = torch.matmul(scores, v)
        return x
