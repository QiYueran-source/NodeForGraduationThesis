# 设置Python路径
import src.utils.set.set_pypath

# 导入redis 
from src.worker.data.redis import REDIS_CONNECTOR
from src.worker.data.redis import REDIS_PREFIX_MANAGER

# 导入数据加载层
from src.worker.data.loader import DATA_LOADER

# 加载数据
data = DATA_LOADER.fetch_data(1997)

# 打印数据
print(data)