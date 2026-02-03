"""
worker工作流程  
1.引入SAVOR创建目录  
2.开启数据线程
3.开始训练
4.推送结果到主机  
"""
# 设置路径
import src.setpath # 设置路径

# 库
import time
import sys
import torch 

# 组件
from src.worker.cache.pool import DATA_CACHE_POOL # 缓存池
from src.worker.data import data_thread # 数据线程
from src.worker.save import SAVER # 保存器
from src.worker.agent import AGENT_DATA_ADAPTER, REWARD_MANAGER # 数据适配器，奖励管理器 


# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[main]')


def start_running():
    """设置运行状态"""  
    DATA_CACHE_POOL.put_running(True)

def start_datathread():
    """启动数据线程"""
    data_thread.data_thread_start()
    while not DATA_CACHE_POOL.contains_train(DATA_CACHE_POOL.get_start_year()):
         time.sleep(1) # 等待1s保证数据加载完成
    logger.debug("数据线程启动完成")  

def train():
    """训练循环"""

def stop():
    # 保存
    SAVER.append_performance_and_reward_snapshot() # 最后一次落盘 
    SAVER.save_model() # 保存模型
    SAVER.save_status() # 保存状态

    # 停止
    data_thread.data_thread_stop()

    # 设置运行状态
    DATA_CACHE_POOL.put_running(False)

    logger.debug("数据线程停止完成")


if __name__ == "__main__":
    try:
        # 启动日志
        logger.info("===========worker启动===========")
        logger.info(f"task_id: {DATA_CACHE_POOL.get_task_id()}")
        logger.info(f"start_year: {DATA_CACHE_POOL.get_start_year()}")
        logger.info(f"N: {DATA_CACHE_POOL.get_N()}")
        logger.info(f"stock_list: {DATA_CACHE_POOL.get_stock_list()}")
        logger.info(f"earliest_year_month: {DATA_CACHE_POOL.get_earliest_year_month()}")
        logger.info(f"train_config: {DATA_CACHE_POOL.get_train_config()}")
        
        start_running() # 设置运行状态 
        start_datathread() # 启动数据线程 
        train() # 开始训练
        
        # 结束
        stop()
        logger.info("===========worker运行结束===========\n\n")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("收到中断信号，停止worker")
        stop()
        logger.info("===========worker停止===========\n\n")
        sys.exit(0)

    except Exception as e:
        logger.error(f"worker运行失败: {e}")
        sys.exit(1)
