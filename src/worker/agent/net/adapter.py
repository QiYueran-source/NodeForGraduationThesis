"""
网络适配器  
通过统一的参数格式访问，调用不同的模型，封装模型内部细节，对外提供统一的接口。  
model_config 结构见 DATA_CACHE_POOL 顶部注释（cate/cuda/opt/clip_grad_norm/dropout/config）。    

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
        # 模型
        self.model:torch.nn.Module = None

        # 加载配置
        self.m = DATA_CACHE_POOL.get_train_config().get('m', 1)
        self.n = DATA_CACHE_POOL.get_train_config().get('n', 1)
        self.mask_len = DATA_CACHE_POOL.get_train_config().get('mask_len', 60)
        self.model_config = DATA_CACHE_POOL.get_train_config().get('model_config', {})

        # 模型配置
        self.cuda = self.model_config.get('cuda', 0)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        self.opt = self.model_config.get('opt', {})
        self.opt_cate = self.opt.get('cate', 0)
        self.opt_lr = self.opt.get('lr', 1e-3)
        self.opt_weight_decay = self.opt.get('weight_decay', 0)
        self.clip_grad_norm = self.model_config.get('clip_grad_norm', 0)
        self.dropout = self.model_config.get('dropout', 0)
        self.config = self.model_config.get('config', {})
        
        # 获取模型
        self._set_model(self.model_config.get('cate', 0))

        # 检查cuda
        self._to_device()

    def _set_model(self, cate:int):
        """
        设置模型
        cate: 模型类别  
        - 0: mlp1
        """
        if cate == 0:
            self.model = MLP(self.n, self.m, self.mask_len, **self.config)
        else:
            raise

    def _to_device(self):
        """
        将模型移动到 self.device（cuda 或 cpu）
        """
        self.model.to(self.device)
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
        return self.model(obs)


    def get_checkpoint(self) -> dict:
        """
        返回可供 safetensors 保存的 state_dict（键为 str，值为 CPU 上的 Tensor）。
        SAVER 可直接用 safetensors.torch.save_file(get_checkpoint(), path) 保存。
        """
        state_dict = self.model.state_dict()
        return {k: v.cpu().clone() for k, v in state_dict.items()}

NET_ADAPTER = NetAdapter()