# 库
from typing import List, Optional, Dict, Tuple

# 自定义组件 
from src.worker.data.redis import REDIS_CONNECTOR,REDIS_PREFIX_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[DataLoaderFetch]')

# 数据加载层
class DataLoaderFetch:
    def __init__(self):
        self.client = REDIS_CONNECTOR.get_client()
        self.redis_prefix_manager = REDIS_PREFIX_MANAGER

    def fetch_data(self, 
        year: int, 
        month: Optional[int] = None, 
        code_list: Optional[List[str]] = None 
    ) -> Dict[Tuple[int, str], any]:
        """
        从redis中获取数据
        :param year: 年份
        :param month: 月份，如果提供则只加载该月份，否则加载所有月份(1-12)
        :param code_list: 股票代码列表，如果提供则只加载这些代码，否则加载所有代码
        :return: 字典，键为(month, code)元组，值为数据
        """
        # 确定要加载的月份列表
        if month is not None:
            months = [month]
        else:
            months = list(range(1, 13))  # 1-12月
        
        # 确定要加载的代码列表
        if code_list is not None:
            codes = code_list
        else:
            # 从Redis中扫描获取所有代码
            codes = self._get_all_codes(year, months)
        
        # 加载数据
        result = {}
        for m in months:
            for code in codes:
                try:
                    # 构建键
                    slice_key = self.redis_prefix_manager.build_train_slice_key(year, m, code)
                    counter_key = self.redis_prefix_manager.build_counter_key(year, m, code)
                    
                    # 获取数据
                    data = self.client.get(slice_key)
                    
                    # 如果数据存在，则存储并自增计数器
                    if data is not None:
                        result[(m, code)] = data
                        self.client.incr(counter_key)
                except Exception as e:
                    logger.warning("加载数据失败 year=%d, month=%d, code=%s: %s", year, m, code, e)
                    continue
        
        return result
    
    def _get_all_codes(self, year: int, months: List[int]) -> List[str]:
        """
        从Redis中扫描获取指定年份和月份的所有代码
        :param year: 年份
        :param months: 月份列表
        :return: 代码列表
        """
        codes_set = set()
        train_prefix = self.redis_prefix_manager.train_prefix
        
        # 对每个月份扫描键
        for month in months:
            month_str = f"{month:02d}"
            # 构建扫描模式: gt:data:train:{year}:{month}:*
            pattern = f"{train_prefix}:{year}:{month_str}:*"
            
            # 使用scan迭代所有匹配的键
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    # 从键中提取代码: gt:data:train:{year}:{month}:{code}
                    parts = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = parts.split(':')
                    if len(parts) >= 5:
                        code = parts[-1]  # 最后一部分是代码
                        codes_set.add(code)
                
                if cursor == 0:
                    break
        
        return sorted(list(codes_set))