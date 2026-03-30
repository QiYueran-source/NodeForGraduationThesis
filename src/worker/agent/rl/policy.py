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
        critic_net = kwargs.pop("critic_net", None)
        if actor_net is not None:
            kwargs.setdefault("features_extractor_class", MLPFeatureExtractor)
            fe_kwargs = kwargs.setdefault("features_extractor_kwargs", {})
            fe_kwargs["actor_net"] = actor_net
        self.critic_net = critic_net
        if "net_arch" not in kwargs:
            # 当提供 critic_net 时：value 网络将由 critic_net 直接给出，不再需要 SB3 的 vf MLP
            kwargs["net_arch"] = dict(pi=[], vf=[] if critic_net is not None else [64, 64])
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

        if self.critic_net is not None:
            self.critic_net.to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.critic_net is None:
            return super().forward(obs, deterministic=deterministic)

        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.critic_net(obs)

        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.critic_net is None:
            return super().evaluate_actions(obs, actions)

        distribution = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic_net(obs)
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        if self.critic_net is None:
            return super().predict_values(obs)
        return self.critic_net(obs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        mean_actions = self.action_net(latent_pi)
        if torch.isnan(mean_actions).any() or torch.isinf(mean_actions).any():
            bad = torch.isnan(mean_actions).any(dim=1) | torch.isinf(mean_actions).any(dim=1)
            bad_idx = bad.nonzero(as_tuple=True)[0].tolist()
            logger.warning(
                "mean_actions 含 NaN/Inf, shape=%s, 异常 batch 索引: %s",
                tuple(mean_actions.shape),
                bad_idx[:20] if len(bad_idx) > 20 else bad_idx,
            )
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
