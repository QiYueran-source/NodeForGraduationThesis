"""
agent 公共数学工具：两段归一化等。
供 net（torch）与 env（numpy）共用。
"""
import numpy as np
import torch

__all__ = ["two_step_normalize", "two_step_normalize_np"]


def two_step_normalize(
    x: torch.Tensor, short_limit: float, filtered: bool = True
) -> torch.Tensor:
    """
    两阶段标准化（torch，用于 net 前向）。
    short_limit 应小于 0。
    1. 将 x 中 <0 的部分归一化到和为 short_limit
    2. 将 >=0 的部分归一化到和为 1 - short_limit
    filtered: 若 True 且 x 全 <0，则将最后一维置为 1。
    x 全为 0 时返回等权。
    """
    x = x.clone()

    if (x == 0).all():
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype) / x.shape[0]

    if filtered and (x < 0).all():
        x[-1] = 1

    short_mask = x < 0
    long_mask = x >= 0
    if short_mask.any():
        x_short = x[short_mask]
        x_short = x_short / x_short.sum() * short_limit
        x[short_mask] = x_short
    if long_mask.any():
        x_long = x[long_mask]
        eps = 0.0
        if x_long.sum() == 0:
            eps = 1e-6
        x_long = x_long / (x_long.sum() + eps) * (1 - short_limit)
        x[long_mask] = x_long

    return x


def two_step_normalize_np(
    x: np.ndarray, short_limit: float, filtered: bool = True
) -> np.ndarray:
    """
    两阶段标准化（numpy，用于 env 等）。
    语义与 two_step_normalize 一致；不修改入参，返回新数组。
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if (x == 0).all():
        return np.ones(x.shape[0], dtype=x.dtype) / x.shape[0]

    if filtered and (x < 0).all():
        x[-1] = 1

    short_mask = x < 0
    long_mask = x >= 0
    if short_mask.any():
        x_short = x[short_mask]
        x_short = x_short / x_short.sum() * short_limit
        x[short_mask] = x_short
    if long_mask.any():
        x_long = x[long_mask]
        eps = 0.0
        if x_long.sum() == 0:
            eps = 1e-6
        x_long = x_long / (x_long.sum() + eps) * (1 - short_limit)
        x[long_mask] = x_long

    return x
