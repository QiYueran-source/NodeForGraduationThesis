"""
强化学习适配器
"""
# 
# 库
from typing import Any, Optional

# 强化学习
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# 组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.env.rolling_env import ROLLING_ENV
from src.worker.agent.net import NET_ADAPTER
from src.worker.save.saver import SAVER
from src.worker.agent.rl.policy import CustomActorCriticPolicy, MLPFeatureExtractor

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="[ReinforcementLearningAdapter]")

"""
reinforcement_config: 强化学习配置（
    cate: 0 PPO, 1 A2C, 2 SAC, 3 TD3
    rl_config: 内部配置 
        - PPO 
            - n_steps: 多少步后PPO更新   
            - n_epochs: 重复多少轮   
            - batch_size: 每次更新的样本数（整除n_steps)    
    opt: 
        lr
        clip_grad_norm
"""
class CustomCallback(BaseCallback):
    def __init__(self, rl_end_year:Optional[int] = None, save_every_n_steps:Optional[int] = None):
        """
        rl_end_year: 强化学习结束年份,None表示不结束  
        save_every_n_steps: 每多少步保存一次模型,None表示不保存  
        """
        super().__init__()

        # 配置
        self.rl_end_year = rl_end_year
        self.save_every_n_steps = save_every_n_steps

        # 步数游标（每步 +1，满 save_every_n_steps 时保存并清零）
        self._step_cursor = 0

    def _on_step(self) -> bool:
        """
        若 current_year > rl_end_year 则返回 False 结束训练；每 save_every_n_steps 步保存一次模型。
        """
        # 首次进入时打一条调试日志
        if self.n_calls == 1:
            logger.debug(f"callback 首次 _on_step, rl_end_year={self.rl_end_year}")
        # 每 1000 步打一条进度（需从 model 取总步数）
        if self.n_calls >= 1 and self.model is not None and getattr(self.model, "num_timesteps", 0) > 0:
            total = self.model.num_timesteps
            if total % 1000 == 0 and total > 0:
                ym = DATA_CACHE_POOL.get_current_year_month()
                logger.info(f"训练步数: {total}, 当前窗口: {ym}")

        # 判断是否结束 
        ym = DATA_CACHE_POOL.get_current_year_month()
        if ym is not None:
            year, month = ym
            if self.rl_end_year is not None and year > self.rl_end_year:
                logger.info(f"强化学习结束, 年份: {year}, 月份: {month}, 保存")
                SAVER.save_record()
                SAVER.save_model()
                SAVER.append_performance_and_reward_snapshot()
                return False

        # 保存状态(每50步)
        if self._step_cursor % 50 == 0 and self._step_cursor >= 50:
            SAVER.save_record()

        # 保存表现和奖励快照(每300步)
        if self._step_cursor % 300 == 0 and self._step_cursor >= 300:
            logger.debug(f"保存表现和奖励快照, 步数: {self._step_cursor}")
            SAVER.append_performance_and_reward_snapshot()

        # 保存模型 
        self._step_cursor += 1
        if self.save_every_n_steps is not None and self._step_cursor >= self.save_every_n_steps and self._step_cursor % self.save_every_n_steps == 0:
            logger.debug(f"保存模型, 步数: {self._step_cursor}")
            SAVER.save_model()

        return True

class ReinforcementLearningAdapter:
    def __init__(self):
        # 环境配置
        ec = DATA_CACHE_POOL.get_env_config() or {}
        self.rl_end_year = ec.get('rl_end_year', None)
        self.save_every_n_steps = ec.get('save_every_n_steps', None)

        # 训练配置 
        tc = DATA_CACHE_POOL.get_train_config() or {}
        self.seed = tc.get('seed', 42)
        self.m = tc.get('m', 1)
        self.n = DATA_CACHE_POOL.get_n() or 1
        self.mask_len = tc.get('mask_len', 60)
        self.rl_end_year = tc.get('rl_end_year', 2024)
        
        # 强化学习配置
        self.reinforcement_config = tc.get('reinforcement_config', {})
        self.rl_cate = self.reinforcement_config.get('cate', 0)
        self.opt = self.reinforcement_config.get('opt', {})
        self.rl_config = self.reinforcement_config.get('rl_config', {})

        # 检查rl_end_year
        self._check_rl_end_year()

        # 设置随机种子
        set_random_seed(self.seed)

        # 算法
        self.rl_algorithm = None

        # 设置算法
        self._set_algorithm()

        # 回调函数
        self.callback = CustomCallback(self.rl_end_year, self.save_every_n_steps)

    def _check_rl_end_year(self):
        """
        检查rl_end_year <= end_year
        """
        end_year = DATA_CACHE_POOL.get_end_year()
        if end_year is not None and self.rl_end_year > end_year:
            logger.warning(f"rl_end_year {self.rl_end_year} > end_year {end_year}, 设置为 end_year")
            self.rl_end_year = end_year

    def _set_ppo(self):
        if self.rl_config.get('n_steps', 2048) % self.rl_config.get('batch_size', 64) != 0:
            logger.warning("n_steps 不是 batch_size 的整数倍，自动对齐")
            self.rl_config['n_steps'] = self.rl_config['n_steps'] // self.rl_config['batch_size'] * self.rl_config['batch_size']

        self.rl_algorithm = PPO(
            policy=CustomActorCriticPolicy,
            env=ROLLING_ENV,
            policy_kwargs=dict(actor_net=NET_ADAPTER.model),
            learning_rate=self.opt.get('lr', 3e-4),
            n_steps=self.rl_config.get('n_steps', 2048),
            batch_size=self.rl_config.get('batch_size', 64),
            n_epochs=self.rl_config.get('n_epochs', 10),
            max_grad_norm=self.opt.get('max_grad_norm', 0.5),
            seed=self.seed
        )
    
    def _set_a2c(self):
        pass 

    def _set_sac(self):
        pass 

    def _set_td3(self):
        pass 

    def _set_algorithm(self):
        if self.rl_cate == 0:
            self._set_ppo()
        elif self.rl_cate == 1:
            self._set_a2c()
        elif self.rl_cate == 2:
            self._set_sac()
        elif self.rl_cate == 3:
            self._set_td3()
        else:
            logger.error("不支持的强化学习算法")
            raise

    @property
    def algorithm(self)->PPO | A2C | SAC | TD3:
        """
        强化学习算法
        """
        return self.rl_algorithm

    def train(self, total_timesteps: int = 900_000_000):
        """
        训练
        total_timesteps: 总步数,默认大数,由回调函数控制  
        """
        logger.info(f"RL train() 开始, total_timesteps={total_timesteps}")
        self.algorithm.learn(total_timesteps=total_timesteps, callback=self.callback)
        logger.info("algorithm.learn() 已返回")



RL_ADAPTER = ReinforcementLearningAdapter()