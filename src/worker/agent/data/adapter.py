"""
数据适配器，使用AgentDataFetcher获取数据，并转换为Agent可以使用的格式  
"""
# 库 
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

        # 当前窗口
        self._current_year_month = self.earliest_year_month  

        # 加载配置
        self._load_config()

        # 加载股票池
        self._sample_portfolios()

        # 生成掩码
        self._generate_mask()

        # 设置当前窗口（不写 pool，保留 Monitor 的 start_year-1 让首次加载从 start_year 开始）
        self._set_current_year_month(sync_to_pool=False)

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f).get('agent_data', {})
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
        seed = self.train_config.get('seed')
        rng = random.Random(seed) if seed is not None else random.Random()
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

    def _set_current_year_month(self, sync_to_pool: bool = True):
        """
        初始化/对齐当前窗口：default=(start_year-1,1) 与 m 对齐取较晚者。
        sync_to_pool=False 时只更新 self._current_year_month，不写 pool（供 __init__ 用，保留 Monitor 的 start_year-1 让首次加载从 start_year 开始）。
        sync_to_pool=True 时同时写回 pool（供 reset() 用，训练开始时同步对齐后的窗口）。
        """
        default_start_year_month = (self.start_year - 1, 1)
        m = self.train_config.get('m', 1)
        earliest_available_year_month = self._roll_year_month(self.earliest_year_month, m - 1)
        if AgentDataAdapter._year_month_greater(earliest_available_year_month, default_start_year_month):
            self._current_year_month = earliest_available_year_month
        else:
            self._current_year_month = default_start_year_month
        if sync_to_pool:
            DATA_CACHE_POOL.put_current_year_month(self._current_year_month[0], self._current_year_month[1])
        
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
        查询的数据是否大于 meta 当前窗口 current_year_month（即查询的数据还未加载）  
        如果大于，则等待，直到查询到数据，检查是否已删除，如果已删除则报错
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
        
        # 判断是否大于当前窗口 current_year_month，是则循环等待
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if current_ym and self._year_month_greater((year, month), current_ym):
            logger.debug(f"数据大于当前窗口，等待数据加载: ({year}, {month}) > {current_ym}")
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

        # 都不在，返回 None（数据不存在）
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
    def win_roll(self)->bool:
        """
        如果 current_year_month < (end_year, 12)，则滚动训练窗口  
            1._current_year_month += 1  
            2.cusor = 0
            3.DATA_CACHE_POOL.put_current_year_month(...)
            4.删除 adapter 中最早一期的训练数据（已滚出窗口），保持内存小
        否则，返回False
        """
        if AgentDataAdapter._year_month_greater((self.end_year, 12), self._current_year_month):
            self._current_year_month = self._roll_year_month(self._current_year_month, 1)
            # 打乱组合顺序，使下一窗口的采样顺序与本月不同，保证多样性
            with self._portfolio_pool_lock:
                random.shuffle(self._portfolio_pool)
                self._portfolio_cursor = 0
            DATA_CACHE_POOL.put_current_year_month(self._current_year_month[0], self._current_year_month[1])

            # 删除最早一期的训练数据（当前窗口为 current ~ current-(m-1)，不再需要 current-m）
            m = self.train_config.get('m', 1)
            ym_to_drop = self._roll_year_month(self._current_year_month, -m)
            with self._data_pool_lock:
                keys_to_drop = [k for k in self._train_data_pool if (k[0], k[1]) == ym_to_drop]
            for (y, mo, code) in keys_to_drop:
                self.delete_train_data(y, mo, code)
            if keys_to_drop:
                logger.info(f'win_roll: 已删除最早训练数据 {ym_to_drop[0]}-{ym_to_drop[1]}, 条数={len(keys_to_drop)}')

            logger.info(f'滚动成功，当前ym:{self._current_year_month[0]}-{self._current_year_month[1]}')
            return True
        else:
            logger.warning('已经达到结束年月，暂停滚动')
            return False
        
    def win_get_current_year_month(self) -> Tuple[int, int]:
        """
        获取训练窗口的当前年月  
        """
        return self._current_year_month

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
        即current_year_month 到 current_year_month - m + 1 的因子的三维tensor  
        根据portfolio，获取m期因子的三维tensor，若不足m期则返回空tensor  
        按 self.mask 掩码：只取 mask[j]==1 对应位置的因子。  
        """
        factors_tensor = []
        for code in portfolio:
            code_factors = []
            for i in range(self.train_config.get('m', 1)):
                year, month = self._roll_year_month(self._current_year_month, -i)
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
        

        
AGENT_DATA_ADAPTER = AgentDataAdapter()

    
        

    
        

