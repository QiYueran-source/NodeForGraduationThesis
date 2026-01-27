"""
agent的数据获取接口  
"""
# 库
import threading
import yaml 
import time 
import datetime as dt 
from typing import Dict, Optional, Tuple, List 

# 自定义组件
from src.worker.cache import DATA_CACHE_POOL

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[AgentDataGet]')

class AgentDataManager:
    def __init__(self):
        # 配置
        self._config = {}  

        # 数据池  
        # 与进程的数据缓存隔离，可以复用  
        # 字典：key为(year,month,code)，value为数据  
        # value为([因子],收益)
        self._data_pool = {}

        # 锁
        self._lock = threading.Lock()

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f).get('agent_data', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 

    def contains(self, year: int, month: int, code: str) -> bool:
        """
        判断数据是否存在agent缓存池   
        """
        with self._lock:
            data = self._data_pool.get((year,month,code),None)
            if data is not None:
                return True
            return False

    def get_data(self, year: int, month: int, code: str) -> Tuple[List[float], float]:
        """
        按year,month,code获取数据  
        1.先判断在不在agent缓存池    
        如果在，返回；否则  
        2.判断在不在worker缓存池  
        如果在，获取，保存到agent缓存池  
        3.如果不在，则判断  
        查询的数据是否大于meta:now_year(即查询的数据还未加载)  
        如果大于，则等待，直到查询到数据  
        4.否则，报错

        """
        # 判断数据是否存在，存在则获取
        if self.contains(year,month,code):
            return self._data_pool.get((year,month,code))
        
        # 判断是否在worker，是则获取
        if DATA_CACHE_POOL.contains(year,month,code):
            data = DATA_CACHE_POOL.get(year,month,code)
            factors, rtr = data[0], data[1]
            self._data_pool[(year,month,code)] = (factors, rtr)
            return factors, rtr
        
        # 判断是否大于now_year,是则循环等待
        now_year = DATA_CACHE_POOL.get_meta('now_year')
        if year > now_year:
            logger.debug(f"数据大于now_year，等待数据加载: {year} > {now_year}")
            # 循环等待 
            start_time = dt.datetime.now()
            while not self.contains(year,month,code):
                time.sleep(self._config.get('retry_delay', 5))
                now = dt.datetime.now()
                if now - start_time > dt.timedelta(seconds=self._config.get('timeout', 300)):
                    logger.error(f"获取数据超时: {year}, {month}, {code}")
                    raise Exception(f"获取数据超时: {year}, {month}, {code}")
            data = DATA_CACHE_POOL.get(year,month,code)
            factors, rtr = data[0], data[1]
            self._data_pool[(year,month,code)] = (factors, rtr)
            return factors, rtr

        # 都不在，报错
        logger.error(f"数据不在agent缓存池，也不在worker缓存池，且大于now_year: {year}, {month}, {code}")
        raise Exception(f"数据不在agent缓存池，也不在worker缓存池，且大于now_year: {year}, {month}, {code}")
        
    def delete_data(self, year:int, month:int, code: str) -> bool:
        """
        删除数据
        """
        with self._lock:
            if self.contains(year,month,code):
                del self._data_pool[(year,month,code)]
                return True
            return False

AGENT_DATA_MANAGER = AgentDataManager()
            

         