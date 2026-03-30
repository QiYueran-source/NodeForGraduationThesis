"""
网络适配器
通过统一的参数格式访问，调用不同的模型，封装模型内部细节，对外提供统一的接口。
配置来自 get_train_config()（随机部分），model_config 等结构见 pool.py 顶部【train_config = 随机】。
设备：有 GPU 则用 cuda，否则 cpu，由运行时自动决定，无需配置。
"""
# 库
import torch
from safetensors.torch import load_file

# 组件
from src.worker.cache import DATA_CACHE_POOL 
from src.worker.agent.net.mlp import MLP, MLP_Critic
from src.worker.agent.net.tcn import TCN, TCN_Critic
from src.worker.agent.net.lstm import LSTM, LSTM_Critic
from src.worker.agent.net.self_define_mlp import SelfDefineMLP, SelfDefineMLP_Critic

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[NetAdapter]')

class NetAdapter:
    def __init__(self):
        # 模型（内部用 _model，对外通过 property model 只读访问）
        self._model: torch.nn.Module = None

        # 加载配置
        tc = DATA_CACHE_POOL.get_train_config() or {}
        self.seed = tc.get('seed', 42)
        self.m = tc.get('m', 1)
        self.n = DATA_CACHE_POOL.get_n() or 1
        self.short_limit = DATA_CACHE_POOL.get_short_limit() or 0.0
        self.mask_len = tc.get('mask_len', 60)
        self.feature_dim = self.mask_len + 1  # 因子维 + 1 维组合收益率，与 env obs 最后一维一致
        self.model_config = tc.get('model_config', {})

        # 模型配置
        self.cate = self.model_config.get('cate', 0)
        self.output_fun_cate = self.model_config.get('output_fun_cate', 0)  # 0=softmax, 1=tanh
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = self.model_config.get('dropout', 0)
        self.config = self.model_config.get('config', {})

        # 设置随机种子（CPU + GPU，保证可复现）
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # 获取模型
        self._set_model()

        self._to_device()

    def _set_model(self):
        """
        设置模型
        cate: 模型类别
        - 0: MLP
        - 1: TCN
        - 2: LSTM
        """
        if self.cate == 0:
            self._model = MLP(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 1:
            self._model = TCN(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 2:
            self._model = LSTM(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 4:
            self._model = SelfDefineMLP(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        else:
            raise ValueError(f"不支持的 model_config.cate: {self.cate}")

    def _to_device(self):
        """
        将模型移动到 self.device（有 GPU 则 cuda，否则 cpu，自动决定）
        """
        self._model.to(self.device)
        if self.device.type == 'cuda':
            logger.info("使用cuda")

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        动作
        obs: 观测 (n, m, feature_dim) 维度 tensor，feature_dim = mask_len + 1
        """
        if obs.dim() != 3 or obs.shape[0] != self.n or obs.shape[1] != self.m or obs.shape[2] != self.feature_dim:
            logger.error(f"观测维度错误，期望 (n={self.n}, m={self.m}, feature_dim={self.feature_dim})，实际 {obs.shape}")
            raise ValueError("obs shape mismatch")
        return self._model(obs)

    def get_checkpoint(self) -> dict:
        """
        返回可供 safetensors 保存的 state_dict（键为 str，值为 CPU 上的 Tensor）。
        SAVER 可直接用 safetensors.torch.save_file(get_checkpoint(), path) 保存。
        """
        state_dict = self._model.state_dict()
        return {k: v.cpu().clone() for k, v in state_dict.items()}

    def load_checkpoint(self, path: str):
        """
        从 safetensors 文件加载权重到当前模型，用于断点增量训练。
        先加载到 CPU 再 load_state_dict，避免 safetensors 直接 device=cuda 在某些进程环境下报 device invalid。
        path: 如 /Node/checkpoint.safetensors
        """
        state_dict = load_file(path, device="cpu")
        self._model.load_state_dict(state_dict, strict=True)
        logger.warning(f"SB3断点失败: 已从 safetensors 加载模型: {path}")

    @property
    def model(self) -> torch.nn.Module:
        """返回模型（只读）。"""
        return self._model

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        调用模型，返回动作
        输入：
        obs: 观测 (n, m, feature_dim) 维度 tensor，feature_dim = mask_len + 1
        输出：
        - action: 动作，(n+1) 维的权重向量，和为 1
        """
        return self._model(obs)
    

NET_ADAPTER = NetAdapter()


class CriticAdapter:
    """
    critic/value 网络适配器：与 NetAdapter 使用同一个 cate/bakcbone，
    仅将网络输出头替换为 1 个不激活的实数（value）。
    """

    def __init__(self):
        self._model: torch.nn.Module = None

        tc = DATA_CACHE_POOL.get_train_config() or {}
        self.seed = tc.get('seed', 42)
        self.m = tc.get('m', 1)
        self.n = DATA_CACHE_POOL.get_n() or 1
        self.short_limit = DATA_CACHE_POOL.get_short_limit() or 0.0
        self.mask_len = tc.get('mask_len', 60)
        self.feature_dim = self.mask_len + 1
        self.model_config = tc.get('model_config', {})

        self.cate = self.model_config.get('cate', 0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = self.model_config.get('dropout', 0)
        self.config = self.model_config.get('config', {})

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._set_model()
        self._to_device()

    def _set_model(self):
        """
        cate: 模型类别（与 actor 相同）：
        - 0: MLP
        - 1: TCN
        - 2: LSTM
        """
        if self.cate == 0:
            self._model = MLP_Critic(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 1:
            self._model = TCN_Critic(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 2:
            self._model = LSTM_Critic(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        elif self.cate == 4:
            self._model = SelfDefineMLP_Critic(
                self.n,
                self.m,
                self.feature_dim,
                self.dropout,
                self.short_limit,
                **self.config
            )
        else:
            raise ValueError(f"不支持的 model_config.cate: {self.cate}")

    def _to_device(self):
        self._model.to(self.device)
        if self.device.type == 'cuda':
            logger.info("使用cuda")

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self._model(obs)

    @property
    def model(self) -> torch.nn.Module:
        return self._model


CRITIC_ADAPTER = CriticAdapter()