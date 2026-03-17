from typing import Tuple, Optional

import numpy as np

from src.worker.agent.data import AGENT_DATA_ADAPTER
from src.worker.agent.env import REWARD_MANAGER
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__, prefix='[MLPDataset]')


def build_supervised_dataset_for_window(
    current_ym: Tuple[int, int],
    m: int,
    portfolio: Tuple[str, ...],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    基于当前窗口 (year, month) 内的 m 期数据构造监督数据集 (X, y)。

    约定：
    - obs 中的因子张量由 AGENT_DATA_ADAPTER.win_get_factors_tensor(portfolio)
      返回，形状为 (n, m, mask_len)，沿着第 2 维为时间维度。
    - 组合收益率序列使用 REWARD_MANAGER.get_portfolio_return_series(year, month, portfolio)
      返回 shape (m,) 的 object 数组，元素为 float 或 None。
      该序列表示从当前期往前的 m 期收益，需要反转为时间正序。

    构造方式（简单近似实现）：
    - 对 i = 0..(m-2):
        X_i = 第 i 期的因子切片 (n, mask_len) 展平；
        y_i = 第 i+1 期的组合收益（若缺失则跳过该样本）。
    - 返回：
        X: [num_samples, input_dim], input_dim = n * mask_len
        y: [num_samples, 1]
    """
    year, month = current_ym

    # 因子张量 (n, m, mask_len)
    t = AGENT_DATA_ADAPTER.win_get_factors_tensor(portfolio)
    if t is None:
        logger.debug(f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} 因子张量为空")
        return None, None
    factors = t.numpy().astype(np.float32)
    if factors.ndim != 3 or factors.shape[1] != m:
        logger.warning(
            f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} "
            f"因子维度异常, shape={factors.shape}, 期望 m={m}"
        )
        return None, None
    n, m_obs, mask_len = factors.shape
    if m_obs < 2:
        logger.debug(
            f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} "
            f"m_obs={m_obs} < 2, 不构建样本"
        )
        return None, None

    # 组合收益率序列（从当前往前 m 期），反转为时间正序
    rtr_series = REWARD_MANAGER.get_portfolio_return_series(year, month, portfolio)
    if rtr_series is None or rtr_series.size == 0:
        logger.debug(
            f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} "
            f"收益序列为空或 size=0"
        )
        return None, None
    # rtr_series: index0 = 上一期 r_{t-1}，index1 = r_{t-2}, ...
    # 为了与 factors 的时间维度对齐（假设也为从过去到当前），这里简单反转。
    rtr_chrono = rtr_series[::-1]

    X_list = []
    y_list = []

    # 使用 0..m-2 期的因子预测下一期收益
    max_idx = min(m_obs, rtr_chrono.size)
    for i in range(max_idx - 1):
        y_val = rtr_chrono[i + 1]
        if y_val is None:
            continue
        # 第 i 期因子切片 (n, mask_len) 展平
        x_slice = factors[:, i, :]  # (n, mask_len)
        x_flat = x_slice.reshape(-1)
        X_list.append(x_flat)
        y_list.append(float(y_val))

    if not X_list:
        logger.debug(
            f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} "
            f"有效样本为 0, max_idx={max_idx}, m_obs={m_obs}"
        )
        return None, None

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    logger.debug(
        f"build_supervised_dataset_for_window: ym={current_ym}, portfolio={portfolio} "
        f"构建样本数={X.shape[0]}, input_dim={X.shape[1]}"
    )
    return X, y
