"""
自定义特征提取器与策略：用端到端 actor 网络 (n,m,mask_len)->(n+1)，无额外特征提取。
"""
# 库
from typing import Any, Optional

from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="[RLPolicy]")

class MLPFeatureExtractor(BaseFeaturesExtractor):
    """
    用给定的 actor_net 将观测直接映射为 (n+1) 维特征，作为策略的均值输入。
    观测已是提取好的特征，不做额外特征提取；actor_net 需满足：
    输入 (batch, n, m, mask_len)，输出 (batch, n+1)。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        actor_net: Optional[nn.Module] = None,
        features_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        n, m, mask_len = (
            int(observation_space.shape[0]),
            int(observation_space.shape[1]),
            int(observation_space.shape[2]),
        )
        if features_dim is None:
            features_dim = n + 1
        super().__init__(observation_space, features_dim)

        if actor_net is None:
            logger.error("MLPFeatureExtractor 需要传入 actor_net (端到端网络).")
            raise 
        self.actor_net = actor_net
        self._n, self._m, self._mask_len = n, m, mask_len

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """observations: (batch, n, m, mask_len) -> (batch, n+1)."""
        return self.actor_net(observations)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    与 MLPFeatureExtractor 配合：features 已是 actor 输出 (n+1)，pi 分支不再加层，
    action_net 设为 Identity，mean_actions = features；仅增加可学习 log_std 与 value 头。
    使用方式：policy_kwargs=dict(actor_net=你的网络, features_extractor_class=MLPFeatureExtractor)。
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ):
        actor_net = kwargs.pop("actor_net", None)
        if actor_net is not None:
            kwargs.setdefault("features_extractor_class", MLPFeatureExtractor)
            fe_kwargs = kwargs.setdefault("features_extractor_kwargs", {})
            fe_kwargs["actor_net"] = actor_net
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[], vf=[64, 64])
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            return
        action_dim = get_action_dim(self.action_space)
        self.action_net = nn.Identity()
        log_std_init = getattr(self, "log_std_init", 0.0)
        self.log_std = nn.Parameter(
            torch.ones(action_dim, device=self.device) * log_std_init
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
