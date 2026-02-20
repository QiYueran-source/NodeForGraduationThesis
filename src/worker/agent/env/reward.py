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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple, List, Optional, Dict

# 组件
from src.worker.cache import DATA_CACHE_POOL
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER  
from src.utils.warn import deprecated

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[RewardCalculator]')

# 表现指标数量与顺序（与 performance / normalized_performance / reward_weights 一致）
NUM_PERFORMANCE_INDICATORS = 5
PERFORMANCE_INDICATOR_KEYS = ('rtr', 'vol', 'sharpe', 'max_drawdown', 'diversification')


class RewardManager:
    def __init__(self):
        # 配置  
        self._config = {}  

        # 表现配置 
        self._performance_config = DATA_CACHE_POOL.get_performance_config() or {}

        # 回看期数 m：vol/sharpe/max_drawdown 的滚动窗口统一为 m，不再使用 performance_config 的 rolling_window / max_drawdown_window
        tc = DATA_CACHE_POOL.get_train_config() or {}
        self._m = int(tc.get('m', 1))

        # 奖励配置（来自 train_config，结构见 pool.py 顶部【train_config = 随机】）
        self._reward_config = tc.get('reward_config', {}) or {}
        self._reward_weights = self._reward_config.get('reward_weights', {})

        # 锁 
        self._record_lock = threading.Lock()
        
        # 记录  
        self._record = {} # 组合表现字典：key为(year,month,portfolio)，value为 decision_weights, performance:[rtr,vol,sharpe,max_drawdown,diversification], normalized_performance, reward   

        # 快照进度 
        self._snapshot_progress = (-1, -1) # 保证第一次快照不为空

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
    def _calculate_weighted_return(self, return_tuple: Tuple[float], decision_weights: Tuple[float]) -> float:
        """计算加权回报率
        输入： 
        - return_tuple: 收益率tuple，最后一个元素为无风险利率    
        - decision_weights: 证券权重tuple（agent决策输出），最后一项为现金    
        输出：加权回报率  
        """
        if len(return_tuple) != len(decision_weights):
            logger.error(f"收益率tuple和decision_weights长度不一致: {len(return_tuple)} != {len(decision_weights)}")
            raise 
        return sum(w * rtr for rtr, w in zip(return_tuple, decision_weights))

    def _calculate_vol(self, return_series: List[float]) -> float:
        """计算波动率
        输入： 
        - return_series: 收益率序列  
        输出：波动率  
        """
        if len(return_series) <= 1:
            logger.warning(f"收益率序列长度小于等于1，无法计算波动率")
            return 0.0
        if len(return_series) < self._m:
            logger.warning(f"收益率序列长度小于波动率窗口期数 m: {len(return_series)} < {self._m}")
        return np.std(return_series)
    
    def _calculate_sharpe_ratio(self, return_series: List[float]) -> float:
        """计算夏普比率
        输入：
        - return_series: 收益率序列
        - risk_free_rate: 无风险利率
        输出：夏普比率
        """
        risk_free_rate = self._performance_config.get('risk_free_rate', 0.02)
        if len(return_series) <= 1:
            logger.warning(f"收益率序列长度小于等于1，无法计算夏普比率")
            return 0.0
        if len(return_series) < self._m:
            logger.warning(f"计算夏普时，收益率序列长度小于窗口期数 m: {len(return_series)} < {self._m}")
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
        if len(return_series) < self._m:
            logger.warning(f"收益率序列长度小于最大回撤窗口期数 m: {len(return_series)} < {self._m}")

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

    def _max_drawdown_row(self, return_row: np.ndarray) -> float:
        """单行收益率序列的最大回撤（用于向量化批量计算）"""
        valid = return_row[~np.isnan(return_row)]
        if len(valid) <= 1:
            return 0.0
        wealth = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in valid:
            wealth *= 1.0 + r
            if wealth > peak:
                peak = wealth
            if peak > 0:
                dd = (peak - wealth) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _calculate_diversification(self, decision_weights: Tuple[float]) -> float:
        """权重分散度：归一化熵，取值 [0,1]，越大越分散。等权为 1，单押为 0。"""
        w = np.asarray(decision_weights, dtype=np.float64)
        if w.size == 0:
            return 0.0
        eps = 1e-12
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            return 0.0
        w = w / s
        n = w.size
        max_entropy = np.log(n) if n > 1 else 1.0
        entropy = -np.sum(w * np.log(w + eps))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # =============== 表现接口 ===============  
    def calc_portfolio_performance(self, year:int, month:int, portfolio: Tuple[str], decision_weights: Tuple[float]):
        """计算并保存表现  
        输入：
        - year: 年份
        - month: 月份
        - portfolio: 组合  
        - decision_weights: 组合权重（agent决策输出）  
        将表现保存在记录中，key为(year,month,portfolio)，value为{decision_weights:组合权重，rtr:回报率，vol:波动率，sharpe:夏普比率，max_drawdown:最大回撤}   

        计算方法：  
        1. 查询window期收益记录，如果存在，获取，否则计算、写入、获取  
        """
        # 初始化
        key = (year, month, portfolio)  
        record = {}

        # 保存组合权重（agent决策）
        record['decision_weights'] = decision_weights
        
        # 计算并保存表现
        performance = []

        # 计算并保存收益
        return_tuple = AGENT_DATA_ADAPTER.win_get_rtr(portfolio)
        if len(return_tuple) != len(decision_weights):
            logger.error(f"收益率tuple和decision_weights长度不一致: {len(return_tuple)} != {len(decision_weights)}")
            raise Exception(f"收益率tuple和decision_weights长度不一致: {len(return_tuple)} != {len(decision_weights)}")
        rtr = self._calculate_weighted_return(return_tuple, decision_weights)
        performance.append(rtr)
        
        # 获取滚动窗口期收益率序列（从当前期 rtr 开始，往前 m 期，窗口长度 = train_config.m）
        portfolio_return_series_for_rolling = [rtr]
        failed_periods = []
        for i in range(1, self._m):
            y, m = AGENT_DATA_ADAPTER._roll_year_month((year, month), -i)
            past = self._record.get((y, m, portfolio), {})
            perf = past.get('performance')
            rtr = perf[0] if perf and len(perf) >= 1 else None
            if rtr is not None:
                portfolio_return_series_for_rolling.append(rtr)
            else:
                failed_periods.append((y, m, portfolio))
        if failed_periods:
            logger.warning(
                "获取收益率序列失败: 共 %d 期 (当前 %s %s)，示例: %s",
                len(failed_periods), year, month, failed_periods[0],
            )

        portfolio_return_series_for_rolling = portfolio_return_series_for_rolling[::-1] # 反转，idx从最早的ym开始  

        # 计算波动率 
        # 波动率取负数，因为波动率越大，奖励越小
        performance.append(-1 * self._calculate_vol(portfolio_return_series_for_rolling))  

        # 计算夏普比率 
        performance.append(self._calculate_sharpe_ratio(portfolio_return_series_for_rolling))  

        # 计算最大回撤 
        # 最大回撤取负数，因为最大回撤越大，奖励越小  
        performance.append(-1 * self._calculate_max_drawdown(portfolio_return_series_for_rolling))

        # 分散度（归一化熵，越大越分散，与奖励方向一致）
        performance.append(self._calculate_diversification(decision_weights))

        # 保存表现
        record['performance'] = performance

        # 保存记录
        with self._record_lock:
            self._record[key] = record
            logger.debug(f"保存记录: {key}, {record}")

    def get_previous_decision_weights(self, year: int, month: int, portfolio: Tuple) -> Optional[Tuple[float, ...]]:
        """获取该 portfolio 在上一期（日历上一月）的 decision_weights；若无记录或含 NaN 则返回 None。"""
        prev_year, prev_month = AGENT_DATA_ADAPTER._roll_year_month((year, month), -1)
        key = (prev_year, prev_month, portfolio)
        with self._record_lock:
            rec = self._record.get(key, {})
        dw = rec.get("decision_weights")
        if dw is None:
            return None
        dw = tuple(dw) if isinstance(dw, list) else dw
        if len(dw) != len(portfolio) + 1:
            return None
        arr = np.asarray(dw, dtype=np.float64)
        if np.any(np.isnan(arr)):
            return None
        return tuple(float(x) for x in dw)

    def get_portfolio_return_series(self, year: int, month: int, portfolio: Tuple[str]) -> np.ndarray:
        """
        获取「已实现」的 m 期组合收益率序列，仅用于 obs 特征，不包含当前期下一期收益，避免信息泄露。
        约定：get_train_data(y,m) 的 rtr 表示 y,m 到下一期的收益；故决策 (year,month) 时只能用
        r_{t-1}, r_{t-2}, ..., r_{t-m}（从 _record 取）。
        顺序：index 0 = 上一期 r_{t-1}，index 1 = 上上期 r_{t-2}，…，index m-1 = 往前第 m 期 r_{t-m}。
        缺失的期在对应位置填 None。
        返回：shape (m,) 的 object 数组，元素为 float 或 None。
        """
        m = self._m
        if m <= 0:
            return np.array([], dtype=object)
        res = np.empty(m, dtype=object)
        for i in range(m):
            y, mo = AGENT_DATA_ADAPTER._roll_year_month((year, month), -(i + 1))
            key = (y, mo, portfolio)
            with self._record_lock:
                rec = self._record.get(key, {})
            perf = rec.get("performance")
            if not perf or len(perf) < 1:
                res[i] = None
            else:
                res[i] = float(perf[0])
        return res

    def normalize_portfolio_performance(self, year:int, month:int, portfolio: Tuple[str]) -> Optional[List[float]]:
        """归一化组合表现（纵向/滚动窗口标准化，便于在线计算奖励）

        使用 performance_config.std_window 作为纵向标准化窗口：从当前 (year, month) 往前取
        std_window 期，收集这些期内所有组合的该指标值，计算均值和标准差后对当前组合做 z-score。
        分母标准差使用 performance_config.std_floor 做下限（默认 0.01），避免 std 过小导致 z-score 爆炸。

        输入：
        - year: 年份
        - month: 月份
        - portfolio: 组合
        输出：归一化表现 [norm_rtr, norm_vol, norm_sharpe, norm_max_drawdown, norm_diversification]，无记录时返回 None
        """
        key = (year, month, portfolio)

        portfolio_performance_record = self._record.get(key, {}).get('performance', [])
        if not portfolio_performance_record or len(portfolio_performance_record) != NUM_PERFORMANCE_INDICATORS:
            logger.warning(f"组合{portfolio}在{year}年{month}月没有记录或表现长度非{NUM_PERFORMANCE_INDICATORS}，无法计算归一化表现")
            return None

        std_window = self._performance_config.get('std_window')
        if std_window is None:
            logger.warning("performance_config.std_window 未配置，纵向标准化使用默认窗口 24")
            std_window = 24
        if std_window < 2:
            logger.warning(f"performance_config.std_window={std_window} 不足2，无法计算标准差，使用 mean=0 std=1")
            mean_list = [0.0] * NUM_PERFORMANCE_INDICATORS
            std_list = [1.0] * NUM_PERFORMANCE_INDICATORS
        else:
            # 纵向：收集过去 std_window 期内所有记录的各指标值（不包含当前 (year, month)）
            performance_list = [[] for _ in range(NUM_PERFORMANCE_INDICATORS)]
            for i in range(1, std_window + 1):
                y, m = AGENT_DATA_ADAPTER._roll_year_month((year, month), -i)
                for k, rec in self._record.items():
                    if (k[0], k[1]) != (y, m):
                        continue
                    perf = rec.get('performance')
                    if perf and len(perf) >= NUM_PERFORMANCE_INDICATORS:
                        for j in range(NUM_PERFORMANCE_INDICATORS):
                            performance_list[j].append(perf[j])

            if any(len(performance_list[j]) < 2 for j in range(NUM_PERFORMANCE_INDICATORS)):
                logger.warning(
                    f"纵向窗口内样本数不足2，各指标样本数: {[len(performance_list[j]) for j in range(NUM_PERFORMANCE_INDICATORS)]}，使用 mean=0 std=1"
                )
                mean_list = [0.0] * NUM_PERFORMANCE_INDICATORS
                std_list = [1.0] * NUM_PERFORMANCE_INDICATORS
            else:
                mean_list = [float(np.mean(performance_list[i])) for i in range(NUM_PERFORMANCE_INDICATORS)]
                std_floor = self._performance_config.get('std_floor', 0.01)
                std_list = [
                    max(float(np.std(performance_list[i]) + 1e-6), std_floor)
                    for i in range(NUM_PERFORMANCE_INDICATORS)
                ]

        normalized_performance = [
            (portfolio_performance_record[i] - mean_list[i]) / std_list[i]
            for i in range(NUM_PERFORMANCE_INDICATORS)
        ]
        # 标准化结果若为 NaN/Inf，用 0 填充，避免传播到 reward 与后续期
        normalized_performance = [
            float(x) if np.isfinite(x) else 0.0 for x in normalized_performance
        ]

        with self._record_lock:
            self._record[key]['normalized_performance'] = normalized_performance

        return normalized_performance

    def calc_portfolio_reward(self, year:int, month:int, portfolio: Tuple[str]) -> float:
        """
        计算组合奖励
        输入：
        - year: 年份
        - month: 月份
        - portfolio: 组合
        输出：奖励

        若 performance_config.classic_utility 为 True，使用博迪效用 U = μ - (A/2)σ²，
        其中 μ=rtr、σ=vol 从 performance 取，A 从 reward_config.A 取（默认 2）；否则按归一化表现加权求和。
        """
        key = (year, month, portfolio)
        portfolio_record = self._record.get(key, {})
        if not portfolio_record:
            logger.warning(f"组合{portfolio}在{year}年{month}月没有记录，无法计算奖励")
            return 0.0

        # 博迪效用分支：U = μ - (A/2)σ²，不跳过归一化（normalized_performance 仍会写入，供下游解析）
        if self._performance_config.get('classic_utility') is True:
            perf = portfolio_record.get('performance', [])
            if len(perf) < 2:
                logger.warning(f"组合{portfolio}在{year}年{month}月 performance 不足 2 项，无法计算博迪效用奖励")
                return 0.0
            rtr = float(perf[0]) if np.isfinite(perf[0]) else 0.0
            vol = float(-perf[1]) if np.isfinite(perf[1]) else 0.0  # 当前存的是 -vol
            A = float(self._reward_config.get('A', 2.0))
            if A < 0:
                A = 2.0
            reward = rtr - (A / 2.0) * (vol ** 2)
            reward = float(reward) if np.isfinite(reward) else 0.0
            with self._record_lock:
                self._record[key]['reward'] = reward
                logger.debug(f"保存奖励(博迪效用): {key}, reward={reward}, rtr={rtr}, vol={vol}, A={A}")
            return reward

        # 默认：按归一化表现加权求和
        normalized_performance = portfolio_record.get('normalized_performance', [])
        if not normalized_performance:
            logger.warning(f"组合{portfolio}在{year}年{month}月没有归一化表现，无法计算奖励")
            return 0.0
        if len(normalized_performance) != NUM_PERFORMANCE_INDICATORS:
            logger.warning(f"normalized_performance 长度非 {NUM_PERFORMANCE_INDICATORS}: {key}")
            return 0.0
        weight_list = [self._reward_weights.get(k, 0.0) for k in PERFORMANCE_INDICATOR_KEYS]
        reward = sum(w * (r if np.isfinite(r) else 0.0) for w, r in zip(normalized_performance, weight_list))

        with self._record_lock:
            self._record[key]['reward'] = reward
            logger.debug(f"保存奖励: {key}, {reward}")

        return reward

    # =============== 弃用接口 ===============  
    @deprecated("use contains_performance for online reward")
    def contains_reward(self, year: int, month: int, portfolios: List[Tuple[str]]) -> bool:
        """判断 (year, month) 下是否已为所有 portfolios 计算过奖励。"""
        if not portfolios:
            return False
        with self._record_lock:
            return all((year, month, tuple(p)) in self._record for p in portfolios)

    @deprecated("use calc_portfolio_performance for online reward")
    def calc_all_performance(self, year:int, month:int, portfolios:List[Tuple[str]], decision_weights_list:List[Tuple[float]], max_workers:Optional[int] = None):
        """
        弃用

        批量计算当前窗口下所有组合的表现（多线程并行）。

        输入：
        - year: 年份
        - month: 月份
        - portfolios: 组合列表，每个元素为 (code1, code2, ...)
        - decision_weights: 组合权重列表（agent决策），与 portfolios 一一对应，每个元素为 (w1, w2, ..., w_cash)
        - max_workers: 并行线程数，None 表示使用默认（min(32, num_portfolios+4)）

        将每个组合的表现写入 self._record，不返回。
        """
        if len(portfolios) != len(decision_weights_list):
            logger.error(f"组合列表和decision_weights列表长度不一致: {len(portfolios)} != {len(decision_weights_list)}")
            raise ValueError(f"组合列表和decision_weights列表长度不一致: {len(portfolios)} != {len(decision_weights_list)}")
        n = len(portfolios)
        workers = max_workers if max_workers is not None else min(32, n + 4)

        def _task(i: int) -> None:
            self.calc_portfolio_performance(year, month, portfolios[i], decision_weights_list[i])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_task, range(n)))

    @deprecated("use normalize_portfolio_performance for online reward")
    def normalized_all_performance(self, year:int, month:int, portfolios:List[Tuple[str]], max_workers:Optional[int] = None):
        """
        弃用

        批量计算当前窗口下所有组合的归一化表现（多线程并行）。

        输入：
        - year: 年份
        - month: 月份
        - portfolios: 组合列表，每个元素为 (code1, code2, ...)
        """
        n = len(portfolios)
        workers = max_workers if max_workers is not None else min(32, n + 4)

        def _task(i: int) -> None:
            self.normalize_portfolio_performance(year, month, portfolios[i])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_task, range(n)))

    @deprecated("use calc_portfolio_reward for online reward")
    def calc_all_reward(self, year:int, month:int, portfolios:List[Tuple[str]], max_workers:Optional[int] = None):
        """
        弃用

        批量计算当前窗口下所有组合的奖励（多线程并行）。

        输入：
        - year: 年份
        - month: 月份
        - portfolios: 组合列表，每个元素为 (code1, code2, ...)
        """
        n = len(portfolios)
        workers = max_workers if max_workers is not None else min(32, n + 4)

        def _task(i: int) -> None:
            self.calc_portfolio_reward(year, month, portfolios[i])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_task, range(n)))
    
    @deprecated("use calc_portfolio_performance for online reward")
    def calc_reward_in_advance(self, year:int, month:int, portfolios:List[Tuple[str]], decision_weights_list:List[Tuple[float]])->bool:
        """
        弃用

        计算提前奖励

        输入：
        - year: 年份
        - month: 月份
        - portfolios: 组合列表，每个元素为 (code1, code2, ...)
        
        """
        # 验证参数
        if len(portfolios) != len(decision_weights_list):
            logger.error(f"组合列表和decision_weights列表长度不一致: {len(portfolios)} != {len(decision_weights_list)}")
            raise 
        
        # 判断是否已计算过奖励
        if self.contains_reward(year, month, portfolios):
            return False
        
        # 计算奖励
        self.calc_all_performance(year, month, portfolios, decision_weights_list)
        self.normalized_all_performance(year, month, portfolios)
        self.calc_all_reward(year, month, portfolios)
        return True

    # =============== 保存接口 ===============
    def _prune_record(self, current_ym: Tuple[int, int]) -> None:
        """
        裁剪 _record：只保留「两个边界取早的那个」之后的数据，避免内存无限增长。
        - reward 边界：current_ym 往前 retain 期（归一化/上一期权重需要），retain = max(m, std_window) + 1。
        - saver 边界：_snapshot_progress（已交过快照的截止），不能删未保存的。
        删除 (y, m) 严格早于 min(keep_after, _snapshot_progress) 的 key。
        """
        tc = DATA_CACHE_POOL.get_train_config() or {}
        m = int(tc.get("m", 1))
        std_window = self._performance_config.get("std_window") or 24
        retain = max(m, std_window) + 1
        keep_after = AGENT_DATA_ADAPTER._roll_year_month(current_ym, -retain)
        yp, mp = self._snapshot_progress
        # 取更早的作为删除线：只删严格早于该线的 (y,m)
        if AGENT_DATA_ADAPTER._year_month_greater((yp, mp), keep_after):
            cutoff = keep_after
        else:
            cutoff = (yp, mp)
        with self._record_lock:
            to_drop = [
                k for k in self._record.keys()
                if AGENT_DATA_ADAPTER._year_month_greater(cutoff, (k[0], k[1]))
            ]
            for k in to_drop:
                del self._record[k]
            if to_drop:
                logger.debug(f"_prune_record: 删除 {len(to_drop)} 条, cutoff={cutoff}")

    def get_incremental_snapshot(self, year: int, month: int) -> Dict[Tuple[int, int, Tuple[str]], Dict]:
        """获取增量快照
        输入：
        - year: 年份
        - month: 月份
        输出：增量记录 dict，key 为 (year, month, portfolio)，value 为 {decision_weights, performance: [rtr, vol, sharpe, max_drawdown, diversification], reward?}

        增量为上一次 _snapshot_progress 到当前 (year, month) 的差集（不包含上次的 year_month）。
        返回前会裁剪 _record（保留期数 = max(m, std_window)+1，且不删未保存的）。
        """
        yp, mp = self._snapshot_progress

        incremental_record_keys = [
            k for k in self._record.keys()
            if AGENT_DATA_ADAPTER._year_month_greater((k[0], k[1]), (yp, mp))
            and not AGENT_DATA_ADAPTER._year_month_greater((k[0], k[1]), (year, month))
        ]
        incremental_record_dict = {k: self._record[k] for k in incremental_record_keys}

        self._snapshot_progress = (year, month)
        self._prune_record((year, month))
        return incremental_record_dict



REWARD_MANAGER = RewardManager()