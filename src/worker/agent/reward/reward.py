"""
计算和保存奖励，根据实际收益和预测收益的差异，计算奖励，使用SB3进行强化学习   
奖励计算方法：  
- 收益率：sum(权重 * 收益率)    
- 波动：roll_std(vol_win, 收益率)   
- 夏普比率：（收益率 - 无风险利率） / 波动   
- 最大回测：max(回测)   
等  
每一个奖励有一个权重，将奖励横向标准化后加权求和   
"""

# 库
import numpy as np  
import torch 
import math  
import time
import datetime as dt 
import threading
import yaml  
import random 
from typing import Any, Tuple, List, Optional

# 组件
from src.worker.cache import DATA_CACHE_POOL
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER  

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[RewardCalculator]')

class RewardManager:
    def __init__(self):
        # 配置  
        self._config = {}  

        # 表现配置 
        self._performance_config = DATA_CACHE_POOL.get_performance_config() or {}

        # 奖励配置 
        self._reward_config = DATA_CACHE_POOL.get_reward_config() or {}

        # 锁 
        
        # 记录  
        self._performance_record = {} # 组合表现字典：key为(year,month,portfolio)，value为{weights:权重，rtr:回报率，vol:波动率，sharpe:夏普比率，max_drawdown:最大回撤}
        self._reward_record = {} # 组合奖励字典：key为(year,month,portfolio)，value为奖励   

        # 验证
        self._validate_reward_weights()  
        
    def _load_config(self):
        pass 

    def _validate_reward_weights(self):
        """验证权重和为1，否则简单归一化"""
        total_weight = sum(self._reward_weights.values())
        if total_weight == 0:
            return
        original_weights = self._reward_weights.copy()
        for key, weight in self._reward_weights.items():
            self._reward_weights[key] = weight / total_weight
        logger.warning(f"奖励权重和为0，简单归一化:{original_weights} -> {self._reward_weights}")

    # =============== 计算奖励方法 ===============
    def _calculate_weighted_return(self, return_tuple: Tuple[float], weights: Tuple[float]) -> float:
        """计算加权回报率
        输入： 
        - return_tuple: 收益率tuple，最后一个元素为无风险利率    
        - weights: 证券权重tuple，最后一项为现金    
        输出：加权回报率  
        """
        if len(return_tuple) != len(weights):
            logger.error(f"收益率tuple和权重tuple长度不一致: {len(return_tuple)} != {len(weights)}")
            raise 
        return sum(weight * rtr for rtr, weight in zip(return_tuple, weights))

    def _calculate_vol(self, return_series: List[float]) -> float:
        """计算波动率
        输入： 
        - return_series: 收益率序列  
        输出：波动率  
        """
        if len(return_series) <= 1:
            logger.warning(f"收益率序列长度小于等于1，无法计算波动率")
            return 0.0
        if len(return_series) < self._performance_config.get('vol_window', 24):
            logger.warning(f"收益率序列长度小于波动率窗口期数: {len(return_series)} < {self._performance_config.get('vol_window', 24)}")
        return np.std(return_series)
    
    def _calculate_sharpe_ratio(self, return_series: List[float], risk_free_rate: float) -> float:
        """计算夏普比率
        输入：
        - return_series: 收益率序列
        - risk_free_rate: 无风险利率
        输出：夏普比率
        """
        if len(return_series) <= 1:
            logger.warning(f"收益率序列长度小于等于1，无法计算夏普比率")
            return 0.0
        if len(return_series) < self._performance_config.get('vol_window', 24):  
            logger.warning(f"计算夏普时，收益率序列长度小于波动率窗口期数: {len(return_series)} < {self._performance_config.get('vol_window', 24)}")
        return (np.mean(return_series) - risk_free_rate) / (np.std(return_series) + 1e-6) # 避免除0   
    
    def _calculate_max_drawdown(self, return_series: List[float]) -> float:
        """计算最大回撤
        输入：
        - return_series: 收益率序列（索引小表示过去的收益）
        输出：最大回撤（非负，0 表示无回撤；仅一期时返回 0）
        """
        if len(return_series) <= 1:
            logger.warning(f"收益率序列长度为1，无法计算最大回撤，返回0")  
            return 0.0
        if len(return_series) < self._performance_config.get('max_drawdown_window', 24):
            logger.warning(f"收益率序列长度小于最大回撤窗口期数: {len(return_series)} < {self._performance_config.get('max_drawdown_window', 24)}")

        # 累计净值：wealth[0] = 1 * (1 + r_0), wealth[i] = wealth[i-1] * (1 + r_i)
        wealth = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in return_series:
            wealth *= 1.0 + r
            if wealth > peak:
                peak = wealth
            if peak > 0:
                dd = (peak - wealth) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd  
    
    # =============== 奖励接口 ===============  
    def calc_performance(self, year:int, month:int, portfolio: Tuple[str], weights: Optional[Tuple[float]] = None):
        """计算并保存表现  
        输入：
        - year: 年份
        - month: 月份
        - portfolio: 组合  
        - weights: 权重  
        将奖励保存在记录中，key为(year,month,portfolio)，value为奖励    

        计算方法：  
        1. 查询window期收益记录，如果存在，获取，否则计算、写入、获取  
        """
        # 初始化
        key = (year, month, portfolio)  
        perf_dict = {}

        # 保存权重
        perf_dict['weights'] = weights
        
        # 计算并保存收益
        return_tuple = AGENT_DATA_ADAPTER.win_get_rtr(portfolio)
        if len(return_tuple) != len(weights):
            logger.error(f"收益率tuple和权重tuple长度不一致: {len(return_tuple)} != {len(weights)}")
            raise 
        perf_dict['rtr'] = self._calculate_weighted_return(return_tuple, weights)
        
        # 获取波动率计算需要的收益率序列
        portfolio_return_series_for_vol = []
        for i in range(self._performance_config.get('vol_window', 24)):
            y,m = AGENT_DATA_ADAPTER._roll_year_month((year, month), -i)
            rtr = self._performance_record.get((y,m,portfolio), {}).get('rtr', None)
            if rtr is not None:
                portfolio_return_series_for_vol.append(rtr)
            else:
                logger.warning(f"获取收益率序列失败: {y}, {m}, {portfolio}")

        portfolio_return_series_for_vol = portfolio_return_series_for_vol[::-1]
        


        


        


   