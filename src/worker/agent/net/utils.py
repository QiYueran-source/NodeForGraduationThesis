"""
神经网络工具  
"""
import torch

def two_step_normalize(x:torch.Tensor, short_limit:float, filtered:bool = True):
    """
    两阶段标准化：  
    其中，short_limit应该小于0  
    1.将x中小于0的部分归一化到short_limit  
    2.将其余部分归一化到 1 - short_limit

    filtered: 是否过过滤 x 全都小于0的情况  
    如果为True，且 x 全都小于0，则将最后一个元素设置为1   

    如果 x 全为0，则返回等权重    
    """
    x = x.clone()

    # 如果 x 全为0，则返回等权重
    if (x==0).all():
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype) / x.shape[0]

    # 使用过滤  
    if filtered and (x < 0).all():
        x[-1] = 1

    # 分割 x 为小于0和大于0的部分  
    short_mask = x < 0
    long_mask = x >= 0
    if short_mask.any():
        x_short = x[short_mask]
        x_short = x_short / x_short.sum() * short_limit
        x[short_mask] = x_short
    if long_mask.any():
        x_long = x[long_mask]
        eps = 0
        if x_long.sum() == 0:
            eps = 1e-6
        x_long = x_long / (x_long.sum() + eps) * (1 - short_limit)
        x[long_mask] = x_long

    return x