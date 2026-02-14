"""
TCN（时间卷积）：将 (n, m, mask_len) 视作通道 n*m、长度 mask_len 的序列，因果卷积后映射到 (n+1)。
"""
import torch
import torch.nn as nn
from typing import Optional, List

from src.worker.agent.utils.math import two_step_normalize


class TCN(nn.Module):
    """
    输入: (..., n, m, mask_len)
    输出: (..., n+1)，和为 1（两段归一化）
    时序维度为 mask_len，通道为 n*m；卷积后对时间维做池化再线性到 n+1。
    """

    def __init__(
        self,
        n: int,
        m: int,
        mask_len: int,
        dropout: Optional[float] = None,
        short_limit: float = 0.0,
        **config: dict,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.mask_len = mask_len
        self.output_dim = n + 1
        self.short_limit = short_limit

        in_channels = n * m
        num_channels: List[int] = config.get("num_channels", [64, 64])
        kernel_size: int = config.get("kernel_size", 3)

        layers: List[nn.Module] = []
        pad = kernel_size // 2  # 保持时间维长度不变
        for i, out_ch in enumerate(num_channels):
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else num_channels[i - 1],
                    out_ch,
                    kernel_size,
                    padding=pad,
                )
            )
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.conv_stack = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        hidden = num_channels[-1]
        self.fc = nn.Linear(hidden, self.output_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n, m, mask_len) -> (batch, n*m, mask_len)
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, self.n * self.m, self.mask_len)

        x = self.conv_stack(x)
        x = self.pool(x)  # (batch, hidden, 1)
        x = x.squeeze(-1)  # (batch, hidden)
        x = self.fc(x)
        x = self.activation(x)
        x = torch.stack([two_step_normalize(x[i], short_limit=self.short_limit) for i in range(x.shape[0])])

        if batch_shape:
            x = x.reshape(*batch_shape, self.output_dim)
        return x
