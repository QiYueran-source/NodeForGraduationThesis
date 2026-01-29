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
def data_thread_start():
    logger.info('数据层线程开始执行')
    DATA_MONITOR.start()
    logger.debug('DATA_MONITOR 已启动，将按 check_interval 定期检查并加载数据到 DATA_CACHE_POOL')

def data_thread_stop():
    logger.info('数据层线程停止执行')
    DATA_MONITOR.stop()
    logger.debug('DATA_MONITOR 已停止')

