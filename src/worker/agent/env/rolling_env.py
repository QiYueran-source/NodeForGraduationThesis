"""
滚动窗口 Gym 环境：一组合一局，obs = (n, m, mask_len)，action = (n+1)。
reset 返回下一组合的观测；step 内在线计算 reward 并结束本局。
"""
# 库
from typing import Tuple, Optional
import numpy as np
import gymnasium as gym

# 组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER
from src.worker.agent.env.reward import REWARD_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="[RollingEnv]")


class RollingEnv(gym.Env):
    """一组合一局：obs=(n,m,mask_len)，action=(n+1)；step 内在线算 reward 并结束本局。"""

    def __init__(self):
        super().__init__()

        # 配置
        tc = DATA_CACHE_POOL.get_train_config() or {}
        ec = DATA_CACHE_POOL.get_env_config() or {}
        self.n = int(DATA_CACHE_POOL.get_n() or 1)  # 每个组合标的数（固定，meta 顶层）
        self.m = int(tc.get("m", 1))  # 回看期数
        self.mask_len = int(tc.get("mask_len", 60))
        self.seed = int(tc.get("seed", 42))
        box_min = float(ec.get("box_min", 0.0))
        box_max = float(ec.get("box_max", 1.0))

        # obs 三维，展平由网络实现
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n, self.m, self.mask_len), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=box_min, high=box_max, shape=(self.n + 1,), dtype=np.float32
        )

        # 占位零观测（本局结束或无组合时返回，只读勿改）
        self._zero_obs = np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        # 当前 obs 对应的组合，下次 step(action) 时用
        self._pending_portfolio = None

    def _get_obs(self, portfolio) -> np.ndarray:
        """返回该组合的因子 (n, m, mask_len)，不展平。"""
        if portfolio is None or len(portfolio) == 0:
            return np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        t = AGENT_DATA_ADAPTER.win_get_factors_tensor(tuple(portfolio))
        if t.numel() == 0:
            return np.zeros((self.n, self.m, self.mask_len), dtype=np.float32)
        obs = t.numpy().astype(np.float32)
        if np.any(np.isnan(obs)):
            logger.warning("obs 中含 NaN, portfolio=%s", portfolio)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境
        输入：
        - seed: 随机种子
        - options: 选项
        
        对一个portfolio的决策视为一局游戏  
        重置，会获取当前窗口的下一个portfolio，并返回其观测  
        如果已经用完，会自动rolling  

        输出：
        - obs: 观测
        - info: 信息
        """
        if seed is None:
            seed = self.seed
        super().reset(seed=seed, options=options)
        np.random.seed(seed)

        # 获取当前窗口的下一个 portfolio
        next_portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()
        if len(next_portfolio) == 0:
            logger.info(f" {AGENT_DATA_ADAPTER.win_get_current_year_month()} 窗口没有可用的portfolio，滚动窗口")
            
            # 滚动窗口
            rolling = AGENT_DATA_ADAPTER.win_roll()

            # 获取下一个portfolio
            next_portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()
            
            # 滚动窗口失败
            if not rolling or len(next_portfolio) == 0:
                logger.warning(f" {AGENT_DATA_ADAPTER.win_get_current_year_month()} 已经到达结束窗口/没有可用的portfolio，返回zero_obs")
                info = {"msg": "no pending portfolio", "no_more_episodes": True}
                return self._zero_obs, info

        self._pending_portfolio = next_portfolio
        obs = self._get_obs(next_portfolio)

        # 获取年月
        year, month = AGENT_DATA_ADAPTER.win_get_current_year_month()
        info = {"year": year, "month": month, "msg": "success"}
        logger.debug(f"reset 返回 obs, year={year}, month={month}, portfolio={self._pending_portfolio}")

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        输入：
        - action: 动作，(n+1)维的权重向量，和为1

        输出：
        - next_obs: 占位零观测（本局结束）
        - reward: 奖励，在线计算
        - terminated: True
        - truncated: False
        - info: 信息
        """
        # 调试阻塞
        import time 
        time.sleep(1200)
        print('调试阻塞1200s')

        year, month = AGENT_DATA_ADAPTER.win_get_current_year_month()
        info = {"year": year, "month": month, "msg": "failed"}

        # action 转换
        if not isinstance(action, np.ndarray):
            logger.warning("action 不是np.ndarray，转换为np.ndarray")
            action = np.asarray(action)
        if action.ndim != 1:
            logger.warning("action 不是一维数组，转换为一维数组")
            action = action.reshape(-1)
        if action.shape[0] != self.n + 1:
            logger.warning("action 长度不为n+1，返回zero_obs")
            info["reason"] = "invalid_action_length"
            return self._zero_obs, 0.0, True, False, info

        # 兜底：action 含 NaN 时，优先用该 portfolio 在 _record 中的上一期权重，否则等权重
        if np.any(np.isnan(action)) and self._pending_portfolio is not None:
            portfolio_tuple = tuple(self._pending_portfolio)
            prev_weights = REWARD_MANAGER.get_previous_decision_weights(year, month, portfolio_tuple)
            if prev_weights is not None and len(prev_weights) == self.n + 1:
                action = np.array(prev_weights, dtype=np.float64)
                logger.warning("action 含 NaN，已替换为该 portfolio 上一期决策")
            else:
                action = np.ones(self.n + 1, dtype=np.float64) / (self.n + 1)
                logger.warning("action 含 NaN，无该 portfolio 上一期决策，已替换为等权重")
        elif np.any(np.isnan(action)):
            action = np.ones(self.n + 1, dtype=np.float64) / (self.n + 1)
            logger.warning("action 含 NaN，已替换为等权重")

        # 用 softmax 归一化，避免 action_sum 为 0 或 NaN 时除零；极端情况 fallback 等权
        action = np.asarray(action, dtype=np.float64)
        action_sum = float(action.sum())
        if action_sum > 1.0 + 1e-3 or action_sum < 1.0 - 1e-3 or not np.isfinite(action_sum):
            logger.warning("action 和不为1或非有限，进行 softmax 归一化")
            x = np.clip(action, -50.0, 50.0)
            exp_x = np.exp(x)
            s = exp_x.sum()
            if s > 1e-10 and np.isfinite(s):
                action = exp_x / s
            else:
                action = np.ones(self.n + 1, dtype=np.float64) / (self.n + 1)
                logger.warning("softmax 分母过小或非有限，已替换为等权重")
        action = tuple(action.tolist())

        # 计算表现
        if self._pending_portfolio is not None and self._pending_portfolio:
            REWARD_MANAGER.calc_portfolio_performance(year, month, self._pending_portfolio, action)
            REWARD_MANAGER.normalize_portfolio_performance(year, month, self._pending_portfolio)
            reward = REWARD_MANAGER.calc_portfolio_reward(year, month, self._pending_portfolio)
        else:
            reward = 0.0

        info = {"year": year, "month": month, "msg": "success"}
        logger.debug(f"step 完成, reward={reward:.4f}, year={year}, month={month}")
        self._pending_portfolio = None

        return self._zero_obs, reward, True, False, info

ROLLING_ENV = RollingEnv()