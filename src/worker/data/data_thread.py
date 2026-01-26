"""
data线程  
1.定时监控  
2.将数据加载到缓存池   
"""  
# 库  
from src.worker.data.monitor import DATA_MONITOR  

# 日志
from src.utils.logger import get_module_logger  
logger = get_module_logger(__name__,'DataThread')

# 主函数
def start():
    logger.info('数据层线程开始执行')
    DATA_MONITOR.start()  

def stop():
    logger.info('数据层线程停止执行')
    DATA_MONITOR.stop()

