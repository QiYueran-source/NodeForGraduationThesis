# 设置Python路径
import src.set_pypath

# 导入redis 
from src.worker.data.redis import REDIS_CONNECTOR
from src.worker.data.redis import REDIS_PREFIX_MANAGER

# 创建redis客户端
redis_client = REDIS_CONNECTOR.get_client()


# 打印redis客户端
print(redis_client.get(REDIS_PREFIX_MANAGER.build_train_slice_key(1997,1,'000001')))