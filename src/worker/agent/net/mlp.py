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
        self.activation = nn.Tanh()

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
        out = torch.stack([two_step_normalize(out[i], short_limit=self.short_limit) for i in range(out.shape[0])])
        return out 

    def backward(self, loss: torch.Tensor):
        """
        反向传播
        """


class MLP_Critic(nn.Module):
    """
    critic/value 网络：结构与 actor 的 MLP backbone 基本一致，
    但输出层改为单神经元、且不做激活/归一化，输出全实数。
    """

    def __init__(
        self,
        n: int,
        m: int,
        mask_len: int,
        dropout: Optional[float] = None,
        short_limit: float = 0.0,  # 保持签名一致，critic 不使用
        **config: dict,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.mask_len = mask_len
        input_dim = n * m * mask_len

        self.linear = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (..., n, m, mask_len) -> (..., n*m*mask_len)
        x = x.reshape(*x.shape[:-3], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        # 输出: (..., 1)
        return self.linear(x)
        