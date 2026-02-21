"""
数据适配器，使用AgentDataFetcher获取数据，并转换为Agent可以使用的格式  
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

# 日志 
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[AgentDataAdapter]')

class AgentDataAdapter:
    def __init__(self):
        # 训练配置：固定项从 meta 顶层读取（get_N 等），随机项从 get_train_config() 读取，结构见 pool.py 顶部。
        self.train_config = DATA_CACHE_POOL.get_train_config() or {}
        self.N = DATA_CACHE_POOL.get_N()  # 总股票数量
        self.start_year = DATA_CACHE_POOL.get_start_year()  # 开始年份
        self.end_year = DATA_CACHE_POOL.get_end_year()  # 结束年份（结束月份固定为 12）
        self.earliest_year_month = DATA_CACHE_POOL.get_earliest_year_month()  # 最早的年份和月份
        self.mask_len = self.train_config.get('mask_len', 60)  # 因子掩码长度
        self.mask = []

        # 配置 
        self._config = {}    

        # 训练数据池  
        # 字典：key为(year,month,code)，value为数据    
        self._train_data_pool = {}   
        self._deleted_train_data_pool = {}  # 字典：key为(year,month,code)，value为True

        # 组合池（先填股票列表供采样；采样后变为组合列表）
        stock_list = DATA_CACHE_POOL.get_stock_list() or []
        self._portfolio_pool: List = list(stock_list)  # 扁平列表供 _sample_portfolios 采样 n 只标的
        self._portfolio_cursor = 0 # 组合池的游标  

        # 锁
        self._data_pool_lock = threading.Lock() # 保护数据池的锁  
        self._portfolio_pool_lock = threading.Lock() # 保护组合池的锁  

        # 采样与 shuffle 种子、同一窗口多轮训练、仅预测年份是否跑满
        env_config = DATA_CACHE_POOL.get_env_config() or {}
        self._sample_and_shuffle_seed = env_config.get('sample_and_shuffle_seed')
        self._retrain_times = int(env_config.get('retrain_times', 1))
        self._retrain_cursor = 0  # 当前窗口已训练轮数，由 win_roll 根据其与 _retrain_times 决定重置还是滚窗
        _rl = env_config.get('rl_end_year')
        self._rl_end_year = int(_rl) if _rl is not None else None  # 仅预测年份（current_ym[0] > 此值）不跑满，直接滚窗
        self._effective_sample_seed = (
            self._sample_and_shuffle_seed
            if self._sample_and_shuffle_seed is not None
            else self.train_config.get('seed', 42)
        )

        # 加载配置
        self._load_config()

        # 加载股票池（用有效种子保证采样顺序可复现）
        random.seed(self._effective_sample_seed)
        self._sample_portfolios()

        # 生成掩码
        self._generate_mask()

        # 设置当前窗口并写入 record（adapter 为唯一写者，无争用）
        self._set_current_year_month()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self._config = data.get('agent_data') or (data.get('agent') or {}).get('data') or {}
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 

    def _sample_portfolios(self):
        """
        采样若干个组合
        """
        n = DATA_CACHE_POOL.get_n() or 1
        max_portfolios_num = DATA_CACHE_POOL.get_max_portfolios_num() or 1000000
        num = min(max_portfolios_num, math.comb(self.N, n))
        rst_set = set()  # 去重
        while len(rst_set) < num:
            rst_set.add(tuple(random.sample(self._portfolio_pool, n)))
        self._portfolio_pool = list(rst_set)
        logger.debug(f"采样{num}个组合完成")

    def _generate_mask(self):
        """
        生成掩码  
        """
        factors_list = DATA_CACHE_POOL.get_factors_list() or []
        n = len(factors_list)
        effective_mask_len = min(self.mask_len, n)
        seed = self.train_config.get('seed', 42)
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        self.mask = [0] * n
        for i in indices[:effective_mask_len]:
            self.mask[i] = 1
        return self.mask

    @staticmethod
    def _roll_year_month(ym: Tuple[int, int], rolling_m: int) -> Tuple[int, int]:
        """
        滚动年月的方法  
        ym: 年月  
        rolling_m: 滚动月数，可与为负数  
        return: 滚动后的年月  
        """
        year, month = ym
        total_month = year * 12 + (month - 1)   # 0-based 月
        total_month += rolling_m
        year = total_month // 12
        month = total_month % 12 + 1
        return (year, month)

    @staticmethod
    def _year_month_greater(ym1: Tuple[int, int], ym2: Tuple[int, int]) -> bool:
        """
        判断年月是否大于  
        ym1: 年月1  
        ym2: 年月2  
        return: 是否大于  
        """
        return ym1[0] > ym2[0] or (ym1[0] == ym2[0] and ym1[1] > ym2[1])

    def _get_zero_fallback(self) -> Tuple[List[float], float]:
        """取不到数据时的全零兜底：(全零因子列表, 0.0 收益)。因子长度与 self.mask 一致。"""
        n = len(self.mask) if self.mask else self.mask_len
        risk_free_rate = DATA_CACHE_POOL.get_performance_config().get('risk_free_rate', 0.00)
        return ([0.0] * n, risk_free_rate)

    def _set_current_year_month(self):
        """
        初始化/对齐当前窗口并写入 record：default=(start_year, 1)，与 earliest_available（满足 m 期 lookback）取较晚者。
        """
        default_start_year_month = (self.start_year, 1)
        m = self.train_config.get('m', 1)
        earliest_available_year_month = self._roll_year_month(self.earliest_year_month, m - 1)
        if AgentDataAdapter._year_month_greater(earliest_available_year_month, default_start_year_month):
            current_year_month = earliest_available_year_month
        else:
            current_year_month = default_start_year_month
        DATA_CACHE_POOL.put_current_year_month(current_year_month[0], current_year_month[1])
        
    # ========== 训练数据接口 ==========
    def contains_train_data(self, year: int, month: int, code: str) -> bool:
        """
        判断数据是否存在agent缓存池   
        """
        with self._data_pool_lock:
            data = self._train_data_pool.get((year,month,code),None)
            if data is not None:
                return True
            return False

    def get_train_data(self, year: int, month: int, code: str) -> Optional[Tuple[List[float], float]]:
        """
        按year,month,code获取数据  
        1.先判断在不在agent缓存池    
        如果在，检查是否已删除，如果已删除则报错；否则返回
        2.判断在不在worker缓存池  
        如果在，获取，检查是否已删除，如果已删除则报错；否则保存到agent缓存池并返回
        3.如果不在，则判断  
        请求的 (year,month) 是否大于 cache 加载进度 current_train_year_month（即数据尚未加载入 cache）  
        若大于则阻塞等待直到数据进入 cache 或超时
        4.否则，返回 None（数据不存在，如窗口不足 m 期）
        """
        key = (year, month, code)
        
        # 首先检查是否已被删除
        if self.is_train_data_deleted(year, month, code):
            logger.error(f"尝试获取已删除的数据: {year}, {month}, {code}")
            raise Exception(f"数据已被删除，无法获取: {year}, {month}, {code}")
        
        # 判断数据是否存在，存在则获取
        if self.contains_train_data(year, month, code):
            return self._train_data_pool.get(key)
        
        # 判断是否在worker，是则获取
        if DATA_CACHE_POOL.contains_train(year, month, code):
            data = DATA_CACHE_POOL.get_train(code, year, month)
            if data:
                # 再次检查是否已被删除（可能在获取过程中被删除）
                if self.is_train_data_deleted(year, month, code):
                    logger.error(f"尝试获取已删除的数据: {year}, {month}, {code}")
                    raise Exception(f"数据已被删除，无法获取: {year}, {month}, {code}")
                factors, rtr = data[0], data[1]
                self._train_data_pool[key] = (factors, rtr)
                return factors, rtr
        
        # 判断是否超出 cache 加载进度，是则循环等待
        cache_ym = DATA_CACHE_POOL.current_train_year_month
        if cache_ym is None or self._year_month_greater((year, month), cache_ym):
            logger.debug(f"数据超出缓存加载进度，等待数据加载: ({year}, {month}) > {cache_ym}")
            # 循环等待 
            start_time = dt.datetime.now() # 阻塞，直到数据加载完成，超时则报错  
            while not DATA_CACHE_POOL.contains_train(year, month, code): 
                # 每次循环，检查是否已被删除  
                if self.is_train_data_deleted(year, month, code):
                    logger.error(f"尝试获取已删除的数据: {year}, {month}, {code}")
                    raise Exception(f"数据已被删除，无法获取: {year}, {month}, {code}")
                time.sleep(self._config.get('retry_delay', 5))
                now = dt.datetime.now()
                if now - start_time > dt.timedelta(seconds=self._config.get('timeout', 300)):
                    if self._config.get('fallback_to_zero_when_unavailable', False):
                        logger.warning(f"获取数据超时，返回全零兜底: {year}, {month}, {code}")
                        return self._get_zero_fallback()
                    logger.error(f"获取数据超时: {year}, {month}, {code}")
                    raise Exception(f"获取数据超时: {year}, {month}, {code}")
            data = DATA_CACHE_POOL.get_train(code, year, month)
            if data:
                # 再次检查是否已被删除（可能在等待过程中被删除）
                if self.is_train_data_deleted(year, month, code):
                    logger.error(f"尝试获取已删除的数据: {year}, {month}, {code}")
                    raise Exception(f"数据已被删除，无法获取: {year}, {month}, {code}")
                factors, rtr = data[0], data[1]
                self._train_data_pool[key] = (factors, rtr)
                return factors, rtr

        # 都不在，返回 None 或全零兜底（数据不存在）
        if self._config.get('fallback_to_zero_when_unavailable', False):
            logger.warning(f"数据不存在，返回全零兜底: {year}, {month}, {code}")
            return self._get_zero_fallback()
        logger.debug(f"数据不存在，返回 None: {year}, {month}, {code}")
        return None
        
    def delete_train_data(self, year:int, month:int, code: str) -> bool:
        """
        删除数据
        删除后，将删除的键添加到_deleted_train_data_pool中
        """
        with self._data_pool_lock:
            key = (year, month, code)
            if key in self._train_data_pool:
                del self._train_data_pool[key]
                # 将删除的键添加到_deleted_train_data_pool
                self._deleted_train_data_pool[key] = True
                logger.debug(f"数据已删除并记录到删除池: {year}, {month}, {code}")
                DATA_CACHE_POOL.put_record('latest_deleted_data', {"year": year, "month": month, "code": code})
                return True
            return False
    
    def is_train_data_deleted(self, year: int, month: int, code: str) -> bool:
        """
        检查数据是否已被删除
        :param year: 年份
        :param month: 月份
        :param code: 股票代码
        :return: 是否已被删除
        """
        with self._data_pool_lock:
            key = (year, month, code)
            return key in self._deleted_train_data_pool
    
    # ========== 组合池接口 ==========
    def get_portfolio_pool(self) -> List[List[str]]:
        """
        获取证券池
        """
        with self._portfolio_pool_lock:
            return self._portfolio_pool

    # ========== 训练窗口接口 ========== 
    def win_roll(self) -> bool:
        """
        在组合用尽时调用。根据 _retrain_cursor 与 _retrain_times 决定：
        - 若 _retrain_cursor < _retrain_times：仅重置组合游标并打乱当前窗口顺序，不滚窗、不删数据，_retrain_cursor += 1，返回 True。
        - 否则：若 current_year_month < (end_year, 12)，则滚动训练窗口（更新 ym、打乱、删最早一期数据），_retrain_cursor = 0，返回 True；否则返回 False。
        仅预测年份（current_ym[0] > rl_end_year）不跑满多轮，组合用尽后直接滚窗。
        """
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if current_ym is None:
            logger.warning("record 中当前窗口未设置，无法滚动/重置")
            return False
        prediction_only = self._rl_end_year is not None and current_ym[0] > self._rl_end_year
        if not prediction_only and self._retrain_cursor < self._retrain_times:
            with self._portfolio_pool_lock:
                self._portfolio_cursor = 0
                ym_int = current_ym[0] * 12 + current_ym[1]
                shuffle_seed = self._effective_sample_seed + ym_int + self._retrain_cursor
                random.Random(shuffle_seed).shuffle(self._portfolio_pool)
            logger.info(
                f" {current_ym} 窗口本轮已用完，第 {self._retrain_cursor + 1}/{self._retrain_times} 轮，重置游标并打乱后继续"
            )
            self._retrain_cursor += 1
            return True
        if not AgentDataAdapter._year_month_greater((self.end_year, 12), current_ym):
            logger.warning('已经达到结束年月，暂停滚动')
            return False
        
        # 滚窗：先算下一窗，仅跨年时落盘 p&r、推进快照进度并保存模型（每年保存一次）
        from src.worker.save.saver import SAVER
        from src.worker.agent.env import REWARD_MANAGER
        next_ym = self._roll_year_month(current_ym, 1)
        if next_ym[0] != current_ym[0]:
            SAVER.append_performance_and_reward_snapshot(segment=True)
            REWARD_MANAGER.advance_snapshot_progress(current_ym[0], current_ym[1])
            logger.info(f"年份切换 {current_ym[0]} -> {next_ym[0]}，保存模型与 RL checkpoint")
            SAVER.save_model(daemon=False)
            SAVER.save_rl_checkpoint(daemon=False)
        
        with self._portfolio_pool_lock:
            ym_int = next_ym[0] * 12 + next_ym[1]
            shuffle_seed = self._effective_sample_seed + ym_int
            random.Random(shuffle_seed).shuffle(self._portfolio_pool)
            self._portfolio_cursor = 0
        DATA_CACHE_POOL.put_current_year_month(next_ym[0], next_ym[1])
        m = self.train_config.get('m', 1)
        ym_to_drop = self._roll_year_month(next_ym, -m)
        with self._data_pool_lock:
            keys_to_drop = [k for k in self._train_data_pool if (k[0], k[1]) == ym_to_drop]
        for (y, mo, code) in keys_to_drop:
            self.delete_train_data(y, mo, code)
        if keys_to_drop:
            logger.info(f'win_roll: 已删除最早训练数据 {ym_to_drop[0]}-{ym_to_drop[1]}, 条数={len(keys_to_drop)}')
        self._retrain_cursor = 0
        if prediction_only:
            logger.info(f" {current_ym} 仅预测窗，直接滚动窗口至 {next_ym[0]}-{next_ym[1]}")
        else:
            logger.info(f" {current_ym} 窗口已训练 {self._retrain_times} 轮，滚动窗口至 {next_ym[0]}-{next_ym[1]}")
        return True
        
    def win_get_current_year_month(self) -> Tuple[int, int]:
        """
        获取训练窗口的当前年月（从 record 读，adapter 为唯一写者）
        """
        ym = DATA_CACHE_POOL.get_current_year_month()
        if ym is None:
            raise Exception("当前窗口未设置 (record.current_year_month)")
        return ym

    def win_get_a_portfolio(self) -> Tuple[str]:
        """
        获取训练窗口的组合  
        加锁，获取第cursor个组合，cursor+=1  
        如果cursor>=len(组合池)，则返回空列表  
        """
        with self._portfolio_pool_lock:
            cursor = self._portfolio_cursor
            if cursor >= len(self._portfolio_pool):
                return ()
            portfolio = self._portfolio_pool[cursor]
            self._portfolio_cursor += 1
            logger.debug(f"获取{self.win_get_current_year_month()}训练窗口的组合: {portfolio}, cursor: {cursor}")
            return tuple(portfolio)

    def win_get_factors_tensor(self, portfolio: Tuple[str]) -> torch.Tensor:
        """
        获取训练窗口的因子   
        即 current_year_month 到 current_year_month - m + 1 的因子的三维 tensor  
        根据 portfolio 获取 m 期因子，若不足 m 期则返回空 tensor。按 self.mask 掩码。  
        """
        current_ym = self.win_get_current_year_month()
        factors_tensor = []
        for code in portfolio:
            code_factors = []
            for i in range(self.train_config.get('m', 1)):
                year, month = self._roll_year_month(current_ym, -i)
                result = self.get_train_data(year, month, code)
                if result is None:
                    return torch.tensor([], dtype=torch.float32)
                factors, rtr = result
                masked_factors = [factors[j] for j in range(len(factors)) if self.mask[j] == 1]
                code_factors.append(masked_factors)
            factors_tensor.append(code_factors)
        t = torch.tensor(factors_tensor, dtype=torch.float32)
        logger.debug(f"获取{self.win_get_current_year_month()}训练窗口的因子,形状: {t.shape}")
        return t
        
    def win_get_rtr(self, portfolio: List[str]) -> Tuple[float]:  
        """
        获取一个n+1维的tuple，第i个元素为portfolio[i]的收益率，最后一个收益率为无风险利率 
        """
        rtr_tuple = []
        for code in portfolio:
            year, month = self.win_get_current_year_month()
            result = self.get_train_data(year, month, code)
            if result is None:
                return ()
            factors, rtr = result
            rtr_tuple.append(rtr)
        perf = DATA_CACHE_POOL.get_performance_config() or {}
        rtr_tuple.append(perf.get('risk_free_rate', 0.02))
        logger.debug(f"获取{self.win_get_current_year_month()}训练窗口的收益率,长度: {len(rtr_tuple)}")
        return tuple(rtr_tuple)

    def win_get_rtr_series(self, portfolio: Tuple[str]) -> np.ndarray:
        """
        获取当前训练窗口下该组合「已实现」的 m 期组合收益率序列，仅用于 obs 特征。
        不包含当前期下一期收益，避免信息泄露。顺序：index 0 = 上一期，index 1 = 上上期，…，index m-1 = 往前第 m 期。
        返回 shape (m,) 的 object 数组，缺失的期在对应位置为 None。
        """
        from src.worker.agent.env.reward import REWARD_MANAGER
        year, month = self.win_get_current_year_month()
        return REWARD_MANAGER.get_portfolio_return_series(year, month, portfolio)


        
AGENT_DATA_ADAPTER = AgentDataAdapter()

    
        

    
        

