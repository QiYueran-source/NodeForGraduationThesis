"""
LSTM：将 (n, m, mask_len) 视作序列长度 mask_len、每步特征 n*m，LSTM 后映射到 (n+1)。
"""
import torch
import torch.nn as nn
from typing import Optional

from src.worker.agent.utils.math import two_step_normalize


class LSTM(nn.Module):
    """
    输入: (..., n, m, mask_len)
    输出: (..., n+1)，和为 1（两段归一化）
    时序维度为 mask_len，每步特征 n*m；取最后时间步的隐藏状态后线性到 n+1。
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

        input_size = n * m
        hidden_size: int = config.get("hidden_size", 64)
        num_layers: int = config.get("num_layers", 1)
        bidirectional: bool = config.get("bidirectional", False)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if dropout and dropout > 0 and num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_size, self.output_dim)
        self.activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n, m, mask_len) -> (batch, seq_len=mask_len, input_size=n*m)
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, self.mask_len, self.n * self.m)

        out, _ = self.lstm(x)  # (batch, mask_len, hidden*dir)
        x = out[:, -1, :]  # 取最后时间步 (batch, hidden*dir)
        if self._dropout is not None:
            x = self._dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        x = two_step_normalize(x, short_limit=self.short_limit)

        if batch_shape:
            x = x.reshape(*batch_shape, self.output_dim)
        return x
