"""
数据缓存池
线程安全的数据缓存，用于Worker进程内线程间通信
"""
# 库
import threading
from typing import Dict, Optional, Tuple, Any, List

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[DataCachePool]')

class DataCachePool:
    """线程安全的数据缓存池"""
    
    def __init__(self):
        """
        初始化缓存池
        """
        # 缓存数据：键为 (code, year, month)，值为数据
        self._cache: Dict[Tuple[str, int, int], Any] = {}
        
        # 线程锁，保护缓存操作
        self._lock = threading.Lock()
        
        logger.info("数据缓存池已创建")
    
    def put(self, code: str, year: int, month: int, data: Any):
        """
        将数据放入缓存池
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :param data: 数据
        """
        key = (code, year, month)
        
        with self._lock:
            if key in self._cache:
                logger.warning("数据已存在，将被覆盖: %s, %s, %s", code, year, month)
            self._cache[key] = data
            logger.debug("数据已放入缓存: %s, %s, %s", code, year, month)
    
    def batch_put(self, items: List[Dict[str, Any]]):
        """
        批量将数据放入缓存池
        :param items: 数据列表，每个元素包含 code, year, month, data 字段
        """
        with self._lock:
            count = 0
            for item in items:
                code = item.get('code')
                year = item.get('year')
                month = item.get('month')
                data = item.get('data')
                
                if not all([code, year, month, data is not None]):
                    logger.warning("批量插入项格式错误，跳过: %s", item)
                    continue
                
                key = (code, year, month)
                if key in self._cache:
                    logger.debug("批量插入：数据已存在，将被覆盖: %s, %s, %s", code, year, month)
                
                self._cache[key] = data
                count += 1
            
            logger.info("批量插入完成，共插入 %d 条数据", count)
    
    def get(self, code: str, year: int, month: int) -> Optional[Any]:
        """
        从缓存池获取数据并删除（剔除）
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :return: 数据，如果不存在返回 None
        """
        key = (code, year, month)
        
        with self._lock:
            if key in self._cache:
                data = self._cache.pop(key)  # 获取并删除
                logger.debug("数据已从缓存取出并删除: %s, %s, %s", code, year, month)
                return data
            else:
                logger.debug("缓存中不存在: %s, %s, %s", code, year, month)
                return None
    
    def contains(self, code: str, year: int, month: int) -> bool:
        """
        检查数据是否存在（不删除）
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :return: 是否存在
        """
        key = (code, year, month)
        with self._lock:
            return key in self._cache
    
    def peek(self, code: str, year: int, month: int) -> Optional[Any]:
        """
        查看数据但不删除
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :return: 数据，如果不存在返回 None
        """
        key = (code, year, month)
        with self._lock:
            return self._cache.get(key)
    
    def clear(self):
        """
        清空缓存池
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("缓存池已清空，清除了 %d 条数据", count)

# 全局实例  
DATA_CACHE_POOL = DataCachePool() 