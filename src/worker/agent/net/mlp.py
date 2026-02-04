"""
MLP（无中间层）：输入 (n, m, mask_len)，输出 (n+1) 经 softmax 的权重向量。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    单层：展平 → Linear → softmax
    输入: (..., n, m, mask_len)
    输出: (..., n+1)，和为 1
    """

    def __init__(self, n: int, m: int, mask_len: int, **config: dict):
        super().__init__()
        self.n = n
        self.m = m
        self.mask_len = mask_len
        self.output_dim = n + 1

        input_dim = n * m * mask_len
        self.linear = nn.Linear(input_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-3], -1)  # (..., n*m*mask_len)
        out = self.linear(x)                # (..., n+1)
        return torch.softmax(out, dim=-1)
