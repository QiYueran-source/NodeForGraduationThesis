"""
MLP（无中间层）：输入 (n, m, mask_len)，输出 (n+1) 经 softmax 的权重向量。
"""
import torch
import torch.nn as nn
from typing import Optional

# 组件
from src.worker.agent.utils.math import two_step_normalize

class MLP(nn.Module):
    """
    单层：展平 → Linear → softmax
    输入: (..., n, m, mask_len)
    输出: (..., n+1)，和为 1
    """

    def __init__(self, n: int, m: int, mask_len: int, dropout: Optional[float] = None, short_limit: float = 0.0, **config: dict):
        super().__init__()
        self.n = n
        self.m = m
        self.mask_len = mask_len
        self.output_dim = n + 1
        self.short_limit = short_limit

        # 线性层
        input_dim = n * m * mask_len
        self.linear = nn.Linear(input_dim, self.output_dim)

        # 激活层
        self.activation = nn.ReLU()

        #  dropout
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-3], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        out = self.linear(x)
        out = self.activation(out)
        out = two_step_normalize(out, short_limit=self.short_limit)
        return out 

    def backward(self, loss: torch.Tensor):
        """
        反向传播
        """
        