"""
数据监控器   
1.定时监控pool中的数据  
2.如果年份不够，则调用loader加载  
"""  
# 库  
from logging import CRITICAL
from re import T
import time 
import threading  
import yaml  

# 组件 
from src.worker.data.loader import DATA_LOADER  
from src.worker.cache import DATA_CACHE_POOL  

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='DataMonitor')

class DataMonitor:
    def __init__(self):
        # 当前年份
        self.now_year = None

        # 标志 
        started = False

        # 配置 
        self._config = {}

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f).get('data_thread', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 
        
        self.now_year = DATA_CACHE_POOL.get_meta('start_year')
        if not self.now_year:
            logger.error("未设置当前年份")
            raise Exception("未设置当前年份")

    def monitor_loop(self):
        """
        监控数据  
        如果池中数据少于等于最小数据，则调用loader加载  
        加载至最大数据   
        如现在加载到了20
        """
        while self.started:
            time.sleep(self._config.get('check_interval', 10))
            self._check_and_load()

    def _check_and_load(self):
        """检查数据"""
        now_year = DATA_CACHE_POOL.get_meta('now_year')
        if not now_year:
            logger.error("未设置当前年份")
            raise Exception("未设置当前年份")
        
        years = DATA_CACHE_POOL.count_years()
        min_year = self._config.get('min_year', 1)  
        max_year = self._config.get('max_year', 2)    
        if years < min_year:
            logger.info(f"年份数量不足，需要加载数据: {years} <= {min_year}")
            need_load_years = max_year - years # 需要加载的年份数量  
            for year in range(now_year + 1, now_year + need_load_years + 1):
                data = DATA_LOADER.fetch_data(year)
                items = [{'code': code, 'year': year, 'month': month, 'data': data} for month, code, data in data.items()]
                DATA_CACHE_POOL.batch_put(items)
                self.now_year = year # 同步year  
                logger.debug(f'{year}数据加载完毕')  

    def start(self):
        if self.started:
            logger.info('数据监控器已经启动')
            return
        self.started = True  
        threading.Thread(
            target = self.monitor_loop,
            daemon = True 
        )

    def stop(self):
        if not self.started:
            logger.info('数据监控线器已关闭')  
            return  
        self.started = False 
        time.sleep(5)
        # 剩余代码  TODO  


DATA_MONITOR = DataMonitor() 
    

    




    