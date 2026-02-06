"""
强化学习适配器
"""
# 
# 库
from typing import Any, Optional

# 强化学习
from stable_baselines3 import PPO

# 组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.rl.policy import CustomActorCriticPolicy, MLPFeatureExtractor
from src.worker.agent.net.adapter import NetAdapter
from src.worker.agent.env.rolling_env import ROLLING_ENV
from src.worker.agent.env.reward import RewardManager

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="[ReinforcementLearningAdapter]")

"""
reinforcement_config: 强化学习配置（
rl_config:内部配置
cate
opt: 
    lr
    clip_grad_norm
    weight_decay
    n_steps
    batch_size
    n_epochs
"""
class ReinforcementLearningAdapter:
    def __init__(self):
        # 配置 
        self.n = DATA_CACHE_POOL.get_n()
        self.seed = DATA_CACHE_POOL.get_seed()
        
        tc = DATA_CACHE_POOL.get_train_config() or {}
        self.m = tc.get('m', 1)
        self.mask_len = tc.get('mask_len', 60)
        self.rl_config = tc.get('rl_config', {})