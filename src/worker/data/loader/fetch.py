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

    def fetch_data(self, year: int, month: int, code: str):
        """
        从redis中获取数据
        :param year: 年份
        :param month: 月份
        :param code: 股票代码
        :return: 数据
        """
        # 构建键
        slice_key = self.redis_prefix_manager.build_train_slice_key(year, month, code)
        counter_key = self.redis_prefix_manager.build_counter_key(year, month, code)

        # 获取数据
        data = self.client.get(slice_key)

        # 自增
        self.client.incr(counter_key)

        return data