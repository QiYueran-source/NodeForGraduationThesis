"""
滚动窗口 Gym 环境：一步 = 一个组合，obs = (n, m, mask_len)，action = (n+1)，展平由网络实现。
同一窗口内缓冲所有组合的 action 后统一算 reward。
"""
# 库
import numpy as np
import gymnasium as gym
from typing import Tuple

# 组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER
from src.worker.agent.env.reward import REWARD_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="[RollingEnv]")


class RollingEnv(gym.Env):
    """一步一个组合：obs=(n,m,mask_len)，action=(n+1)；窗口内缓冲 action 后统一算 reward。"""

    def __init__(self):
        super().__init__()
        tc = DATA_CACHE_POOL.get_train_config() or {}
        self.n = int(tc.get("n", 1))  # 每个组合标的数
        self.m = int(tc.get("m", 1))  # 回看期数
        self.mask_len = int(tc.get("mask_len", 60))
        self.seed = int(tc.get("seed", 42))

        # obs 三维，展平由网络实现
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n, self.m, self.mask_len), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n + 1,), dtype=np.float32
        )

        # 一步一个组合：缓冲当前窗口的 (portfolio, action)，下一组合由 _pending_portfolio 表示
        self._buffer: list = []  # [(portfolio_tuple, weights_tuple), ...]
        self._pending_portfolio = None  # 当前 step 返回的 obs 对应的组合，下次 step(action) 时用

    def _get_obs(self, portfolio) -> np.ndarray:
        """返回该组合的因子 (n, m, mask_len)，不展平。"""
        if portfolio is None or len(portfolio) == 0:
            return np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        t = AGENT_DATA_ADAPTER.win_get_factors_tensor(tuple(portfolio))
        if t.numel() == 0:
            return np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        return t.numpy().astype(np.float32)

    def reset(self, options=None):
        np.random.seed(self.seed)
        self._buffer = []
        AGENT_DATA_ADAPTER._set_current_year_month()
        AGENT_DATA_ADAPTER._portfolio_pool_cursor = 0
        DATA_CACHE_POOL.put_current_year_month(
            AGENT_DATA_ADAPTER._current_year_month[0], AGENT_DATA_ADAPTER._current_year_month[1]
        )
        self._pending_portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()
        obs = self._get_obs(self._pending_portfolio)
        if len(self._pending_portfolio) == 0:
            self._pending_portfolio = None
        info = {
            "year": AGENT_DATA_ADAPTER._current_year_month[0],
            "month": AGENT_DATA_ADAPTER._current_year_month[1],
        }
        return obs, info

    def step(self, action)->Tuple[np.ndarray, float, bool, bool, dict]:
        """
        输入：
        - action: 动作，(n+1)维的权重向量，和为1  
        输出：
        - next_obs: 下一个观测，(n, m, mask_len)  
        - reward: 奖励  
        - terminated: 是否终止  
        - truncated: 是否截断  
        - info: 信息  
        """
        zero_obs = np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        if self._pending_portfolio is None or len(self._pending_portfolio) == 0:
            return zero_obs, 0.0, True, False, {"msg": "no pending portfolio"}

        # 当前 obs 对应的组合的 action
        w = np.asarray(action, dtype=np.float32).flatten()

        # 权重恢复(如果权重和不为1，则归一化, 允许1e-6误差)
        if w.sum() > 1 + 1e-6 or w.sum() < 1 - 1e-6:
            w = np.clip(w, 1e-6, 1.0)
            w = w / w.sum()

        # 将当前组合和权重加入缓冲区
        self._buffer.append((tuple(self._pending_portfolio), tuple(float(x) for x in w)))

        # 下一个组合
        next_portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()

        if len(next_portfolio) == 0:
            # 当前窗口所有组合已收集完，算 reward 并滚动
            portfolios = [p for p, _ in self._buffer]
            weights_list = [a for _, a in self._buffer]
            year, month = AGENT_DATA_ADAPTER.win_get_current_year_month()
            REWARD_MANAGER.calc_all_performance(year, month, portfolios, weights_list)
            REWARD_MANAGER.normalized_all_performance(year, month, portfolios)
            REWARD_MANAGER.calc_all_reward(year, month, portfolios)
            rewards = [
                REWARD_MANAGER._record.get((year, month, p), {}).get("reward", 0.0)
                for p in portfolios
            ]
            reward = float(np.mean(rewards)) if rewards else 0.0

            self._buffer = [] # 清空缓冲区

            # 滚动训练窗口
            rolled = AGENT_DATA_ADAPTER.win_roll() 
            if rolled:
                self._pending_portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()
                if len(self._pending_portfolio) == 0:
                    self._pending_portfolio = None
                    next_obs = zero_obs
                    terminated = True
                else:
                    next_obs = self._get_obs(self._pending_portfolio)
                    terminated = False
            else:
                self._pending_portfolio = None
                next_obs = zero_obs
                terminated = True
            info = {"year": year, "month": month, "reward": reward}
            return next_obs, reward, terminated, False, info

        # 同一窗口内下一组合，reward 延后
        self._pending_portfolio = next_portfolio
        next_obs = self._get_obs(next_portfolio)
        reward = 0.0
        terminated = False
        info = {
            "year": AGENT_DATA_ADAPTER._current_year_month[0],
            "month": AGENT_DATA_ADAPTER._current_year_month[1],
        }
        return next_obs, reward, terminated, False, info
