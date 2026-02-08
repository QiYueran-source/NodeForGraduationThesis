"""
网络适配器
通过统一的参数格式访问，调用不同的模型，封装模型内部细节，对外提供统一的接口。
配置来自 get_train_config()（随机部分），model_config 等结构见 pool.py 顶部【train_config = 随机】。
设备：有 GPU 则用 cuda，否则 cpu，由运行时自动决定，无需配置。
"""
# 库
import torch

# 组件
from src.worker.cache import DATA_CACHE_POOL 
from src.worker.agent.net.mlp import MLP

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
        self.mask_len = tc.get('mask_len', 60)
        self.model_config = tc.get('model_config', {})

        # 模型配置
        self.cate = self.model_config.get('cate', 0)
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
        - 0: mlp1
        """
        if self.cate == 0:
            self._model = MLP(self.n, self.m, self.mask_len, self.dropout, **self.config)
        else:
            raise

    def _to_device(self):
        """
        将模型移动到 self.device（有 GPU 则 cuda，否则 cpu，自动决定）
        """
        self._model.to(self.device)
        if self.device.type == 'cuda':
            logger.info("使用cuda")

    def act(self, obs:torch.Tensor)->torch.Tensor:
        """
        动作
        obs: 观测(m,n,mask_len)维度tensor  
        """
        if obs.dim() != 3 or obs.shape[0] != self.m or obs.shape[1] != self.m or obs.shape[2] != self.mask_len:
            logger.error(f"观测维度错误，期望{(self.m,self.n,self.mask_len)}，实际{obs.shape}")
            raise 
        return self._model(obs)

    def get_checkpoint(self) -> dict:
        """
        返回可供 safetensors 保存的 state_dict（键为 str，值为 CPU 上的 Tensor）。
        SAVER 可直接用 safetensors.torch.save_file(get_checkpoint(), path) 保存。
        """
        state_dict = self._model.state_dict()
        return {k: v.cpu().clone() for k, v in state_dict.items()}
    
    @property
    def model(self) -> torch.nn.Module:
        """返回模型（只读）。"""
        return self._model

    def __call__(self, obs:torch.Tensor)->torch.Tensor:
        """
        调用模型，返回动作  
        输入：  
        obs: 观测(m,n,mask_len)维度tensor     
        输出： 
        - action: 动作，(n+1)维的权重向量，和为1    
        """
        return self._model(obs)
    

NET_ADAPTER = NetAdapter()