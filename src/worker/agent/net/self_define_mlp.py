"""
self_define_mlp：按 cate=4 的自定义决策网络

输入：obs (n, m, feature_dim)，其中 feature_dim = factor_dim + 1，
obs 最后一维最后一个元素为该期组合收益(用于收益序列)。

逻辑：
- 只取当期(回看期视为 1)：因子取 index=0
- 收益取过去 m-1 期：收益取 index=1..m-1
- 将「当期因子」+「过去 m-1 期收益」拼成一维向量
- 隐藏层：hidden_size=h
- 输出：单神经元，经过 sigmoid 并线性放缩到决策范围
  w = sigmoid(x) * (1 - 2L) + L   (L=short_limit)
- 无风险现金：cash = 1 - w
- 决策权重：decision_weights = [w, cash]

注意：
- 当前实现假设 n=1（action_space 为 n+1=2），否则输出维度会与环境动作维不匹配。
"""

from typing import Optional

import torch
import torch.nn as nn


class SelfDefineMLP(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        mask_len: int,
        dropout: Optional[float] = None,
        short_limit: float = 0.0,
        **config: dict,
    ):
        """
        mask_len 实际上等于 NetAdapter 传入的 feature_dim = 原始因子维 + 1。
        因此真实因子维 factor_dim = mask_len - 1。
        """
        super().__init__()
        if n != 1:
            raise ValueError(f"SelfDefineMLP 目前仅支持 n=1，实际 n={n}")

        self.n = n
        self.m = m
        self.feature_dim = mask_len
        self.factor_dim = mask_len - 1  # 最后一维是收益标量
        if self.factor_dim <= 0:
            raise ValueError(f"feature_dim={mask_len} 不合法，factor_dim=mask_len-1={self.factor_dim}")

        self.short_limit = float(short_limit)

        hidden_size: int = int(config.get("hidden_size", 64))

        # 拼接向量维度 = 因子(当期) + 收益(过去 m-1 期)
        self.rtr_dim = max(self.m - 1, 0)
        input_dim = self.factor_dim + self.rtr_dim

        self.hidden = nn.Linear(input_dim, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else None

        self.out = nn.Linear(hidden_size, 1)  # 单神经元输出 logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n, m, feature_dim) 或 (n, m, feature_dim) 取决于上层是否做 batch 展开
        返回：
        - (B, 2) 或 (2,)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # x: (B, 1, m, feature_dim)
        # 当期因子：index=0，取前 factor_dim
        factor_latest = x[:, :, 0, : self.factor_dim]  # (B, 1, factor_dim)
        factor_latest_flat = factor_latest.reshape(x.shape[0], -1)  # (B, factor_dim)

        # 过去 m-1 期收益：index=1..m-1，收益在最后一维
        if self.rtr_dim > 0:
            rtr_seq = x[:, :, 1:, self.factor_dim]  # (B, 1, m-1)
            rtr_flat = rtr_seq.reshape(x.shape[0], -1)  # (B, m-1)
            vec = torch.cat([factor_latest_flat, rtr_flat], dim=-1)  # (B, factor_dim+(m-1))
        else:
            vec = factor_latest_flat

        h = self.hidden(vec)
        h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)

        logit = self.out(h)  # (B, 1)
        prob = torch.sigmoid(logit)  # (B, 1)

        # w in [L, 1-L]；当 L < 0 时等价于 [-L, 1+L]
        L = self.short_limit
        w = prob * (1.0 - 2.0 * L) + L  # (B, 1)
        cash = 1.0 - w  # (B, 1)

        decision_weights = torch.cat([w, cash], dim=-1)  # (B, 2)

        if squeeze_back:
            decision_weights = decision_weights.squeeze(0)
        return decision_weights


class SelfDefineMLP_Critic(nn.Module):
    """
    critic/value 版本：同一个输入向量，但输出一个不激活的标量 value。
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
        if n != 1:
            raise ValueError(f"SelfDefineMLP_Critic 目前仅支持 n=1，实际 n={n}")

        self.n = n
        self.m = m
        self.feature_dim = mask_len
        self.factor_dim = mask_len - 1
        if self.factor_dim <= 0:
            raise ValueError(f"feature_dim={mask_len} 不合法，factor_dim=mask_len-1={self.factor_dim}")

        hidden_size: int = int(config.get("hidden_size", 64))
        self.rtr_dim = max(self.m - 1, 0)
        input_dim = self.factor_dim + self.rtr_dim

        self.hidden = nn.Linear(input_dim, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else None
        self.out = nn.Linear(hidden_size, 1)  # 直接线性 value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        factor_latest = x[:, :, 0, : self.factor_dim]  # (B, 1, factor_dim)
        factor_latest_flat = factor_latest.reshape(x.shape[0], -1)  # (B, factor_dim)

        if self.rtr_dim > 0:
            rtr_seq = x[:, :, 1:, self.factor_dim]  # (B, 1, m-1)
            rtr_flat = rtr_seq.reshape(x.shape[0], -1)  # (B, m-1)
            vec = torch.cat([factor_latest_flat, rtr_flat], dim=-1)
        else:
            vec = factor_latest_flat

        h = self.hidden(vec)
        h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)

        value = self.out(h)  # (B, 1)
        if squeeze_back:
            value = value.squeeze(0)
        return value

