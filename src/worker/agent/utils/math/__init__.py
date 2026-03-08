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
    正确逻辑：保持相对权重比例，只在超出限制时才按比例缩小。
    filtered: 若 True 且 x 全 <0，则将最后一维置为 |short_limit|。
    x 全为 0 时返回等权。
    """
    x = x.clone()

    if (x == 0).all():
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype) / x.shape[0]

    # 处理全为负数的情况
    if filtered and (x < 0).all():
        x[-1] = abs(short_limit)  # 设置为合理的正值

    # 分离做空和做多部分
    short_mask = x < 0
    long_mask = x >= 0

    short_sum = x[short_mask].sum() if short_mask.any() else 0.0
    long_sum = x[long_mask].sum() if long_mask.any() else 0.0

    # 根据做空是否超出限制决定处理方式
    if abs(short_sum) <= abs(short_limit):
        # 做空在限制内：保持做空部分不变，做多部分补上释放的资金
        target_long_sum = 1.0 + short_sum
        if long_mask.any() and long_sum > 0:
            x[long_mask] = x[long_mask] / long_sum * target_long_sum
    else:
        # 做空超出限制：按比例缩小做空部分，做多部分获得满额释放资金
        if short_mask.any() and short_sum < 0:  # 确保有做空且和为负
            x[short_mask] = x[short_mask] / short_sum * short_limit
        target_long_sum = 1.0 - short_limit
        if long_mask.any() and long_sum > 0:
            x[long_mask] = x[long_mask] / long_sum * target_long_sum
        elif long_mask.any() and long_sum == 0:
            # 做多部分为0的情况，需要给做多部分分配权重
            x[long_mask] = target_long_sum / long_mask.sum()

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

    # 处理全为负数的情况
    if filtered and (x < 0).all():
        x[-1] = abs(short_limit)  # 设置为合理的正值

    # 分离做空和做多部分
    short_mask = x < 0
    long_mask = x >= 0

    short_sum = x[short_mask].sum() if short_mask.any() else 0.0
    long_sum = x[long_mask].sum() if long_mask.any() else 0.0

    # 根据做空是否超出限制决定处理方式
    if abs(short_sum) <= abs(short_limit):
        # 做空在限制内：保持做空部分不变，做多部分补上释放的资金
        target_long_sum = 1.0 + short_sum
        if long_mask.any() and long_sum > 0:
            x[long_mask] = x[long_mask] / long_sum * target_long_sum
    else:
        # 做空超出限制：按比例缩小做空部分，做多部分获得满额释放资金
        if short_mask.any() and short_sum < 0:  # 确保有做空且和为负
            x[short_mask] = x[short_mask] / short_sum * short_limit
        target_long_sum = 1.0 - short_limit
        if long_mask.any() and long_sum > 0:
            x[long_mask] = x[long_mask] / long_sum * target_long_sum
        elif long_mask.any() and long_sum == 0:
            # 做多部分为0的情况，需要给做多部分分配权重
            x[long_mask] = target_long_sum / long_mask.sum()

    return x
