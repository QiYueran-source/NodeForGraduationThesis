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
        self._cache: Dict[Tuple[str, int, int], Any] = {} # 缓存数据：键为 (code, year, month)，值为数据
        self._meta: Dict[str:Any] = {} # 元数据字典
        
        # 线程锁，保护缓存操作
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._meta: Dict[str:Any] = {} # 元数据字典
        
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
    
    def contains(self, year: int, month: Optional[int] = None, code: Optional[str] = None) -> bool:
        """
        检查数据是否存在（支持部分匹配）
        :param year: 年份
        :param month: 月份，默认为None
        :param code: 股票代码，默认为None
        :return: 是否存在符合条件的数据

        匹配逻辑：
        - 如果month和code都提供：检查特定(year, month, code)是否存在
        - 如果只有month提供：检查该年份该月份是否有任何股票数据
        - 如果month和code都为None：检查该年份是否有任何数据
        """
        with self._lock:
            for key in self._cache.keys():
                cache_code, cache_year, cache_month = key

                # 必须匹配年份
                if cache_year != year:
                    continue

                # 如果指定了月份，必须匹配月份
                if month is not None and cache_month != month:
                    continue

                # 如果指定了代码，必须匹配代码
                if code is not None and cache_code != code:
                    continue

                # 找到匹配的数据
                return True

            return False
    
    def count_years(self) -> int:
        """
        获取年份数量
        :return: 年份数量
        """
        with self._lock:
            return len(set([year for _, year, _ in self._cache.keys()]))
    
    def put_meta(self, key: str, value: Any):
        """
        将元数据放入缓存池
        :param key: 键
        :param value: 值
        """
        with self._meta_lock:
            self._meta[key] = value

        
    
    def get_meta(self, key: str) -> Optional[Any]:
        """
        从缓存池获取元数据
        :param key: 键
        :return: 值
        """
        with self._meta_lock:
            return self._meta.get(key, None)
    
    def delete_meta(self, key: str):
        """
        从缓存池删除元数据
        :param key: 键
        """
        with self._meta_lock:
            self._meta.pop(key, None)
    
# 全局实例  
DATA_CACHE_POOL = DataCachePool() 