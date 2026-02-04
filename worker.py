"""
worker工作流程
1. 从命令行参数解析 meta 并写入 DATA_CACHE_POOL
2. 开启数据线程
3. 开始训练
4. 推送结果到主机
"""
# 设置路径
import src.setpath  # 设置路径

# 库
import threading
import os
import argparse
import json
import time
import sys
from pathlib import Path
import torch

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[main]')

# 解析命令行参数，写入 DATA_CACHE_POOL
from src.worker.cache.pool import DATA_CACHE_POOL  # 缓存池
def parse_args_and_load_pool():
    """
    解析命令行参数，做类型转换后写入 DATA_CACHE_POOL。
    list/dict 类参数以 JSON 字符串传入，此处解析；earliest_year_month 转为 tuple。
    """
    parser = argparse.ArgumentParser(description='worker')
    parser.add_argument('--task_id', required=True, help='任务 id')
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--stock_list', required=True, help='JSON 数组字符串，如 ["a","b"]')
    parser.add_argument('--earliest_year_month', required=True, help='JSON 数组 [year, month]')
    parser.add_argument('--train_config', required=True, help='JSON 对象字符串（仅随机部分，结构见 pool.py 顶部）')
    parser.add_argument('--n', type=int, required=True, help='一个组合中的证券数量（固定，顶层）')
    parser.add_argument('--max_portfolios_num', type=int, required=True, help='可构建组合数上限（固定，顶层）')
    parser.add_argument('--performance_config', required=True, help='JSON 对象，表现计算配置（固定，顶层）')
    parser.add_argument('--env_config', required=True, help='JSON 对象，环境配置（固定，顶层）')

    args = parser.parse_args()

    stock_list = json.loads(args.stock_list)
    earliest = json.loads(args.earliest_year_month)
    if isinstance(earliest, list):
        earliest = tuple(earliest)
    train_config = json.loads(args.train_config)

    # 写入 meta 缓存池（结构见 pool.py 顶部：顶层固定 + train_config 随机；end_month 在 put_end_year 内固定为 12）
    DATA_CACHE_POOL.put_task_id(args.task_id)
    DATA_CACHE_POOL.put_start_year(args.start_year)
    DATA_CACHE_POOL.put_end_year(args.end_year)
    DATA_CACHE_POOL.put_N(args.N)
    DATA_CACHE_POOL.put_stock_list(stock_list)
    DATA_CACHE_POOL.put_earliest_year_month(*earliest)
    DATA_CACHE_POOL.put_n(args.n)
    DATA_CACHE_POOL.put_max_portfolios_num(args.max_portfolios_num)
    DATA_CACHE_POOL.put_performance_config(json.loads(args.performance_config))
    DATA_CACHE_POOL.put_env_config(json.loads(args.env_config))
    DATA_CACHE_POOL.put_train_config(train_config)

try:
    parse_args_and_load_pool()
except Exception as e:
    logger.error("解析参数失败: %s", e)
    sys.exit(1)


# 其他组件
from src.worker.data import data_thread  # 数据线程
from src.worker.save import SAVER  # 保存器
from src.worker.agent import AGENT_DATA_ADAPTER, NET_ADAPTER, ROLLING_ENV  # 数据适配器

def start():
    # 设置运行状态
    DATA_CACHE_POOL.put_running(True)
    DATA_CACHE_POOL.put_pid(os.getpid())

    # 初始保存meta、record数据
    SAVER.save_meta() # 保存meta数据   
    SAVER.save_record() # 保存record数据  

    # 启动数据线程 
    data_thread.data_thread_start()  # 启动数据线程 
    time.sleep(10)  # 等待10s保证数据加载完成  

    logger.info("数据线程启动完成，开始训练")


def train():
    """使用NET_ADAPTER,ENV和RL_ADAPTER进行训练"""
    
    
    
def stop():
    # 保存
    SAVER.append_performance_and_reward_snapshot()  # 最后一次落盘
    SAVER.save_model()  # 保存模型
    SAVER.save_status()  # 保存状态（异步写 record.json）

    # 停止
    data_thread.data_thread_stop()

    # 设置运行状态
    DATA_CACHE_POOL.put_running(False)

    logger.debug("数据线程与 record 线程停止完成")


if __name__ == "__main__":
    try:
        logger.info("===========worker启动===========")
        logger.info("task_id: %s", DATA_CACHE_POOL.get_task_id())
        logger.info("start_year: %s", DATA_CACHE_POOL.get_start_year())
        logger.info("N: %s", DATA_CACHE_POOL.get_N())
        logger.info("stock_list: %s", DATA_CACHE_POOL.get_stock_list())
        logger.info("earliest_year_month: %s", DATA_CACHE_POOL.get_earliest_year_month())
        logger.info("train_config: %s", DATA_CACHE_POOL.get_train_config())
        
        print(DATA_CACHE_POOL.get_meta()) # 调试一下  
        
        # 开始运行
        start()  

        # 训练线程
        train_thread = threading.Thread(target=train, daemon=True) 
        train_thread.start() # 开始训练线程
        train_thread.join() # 等待训练线程结束
        exit_code = 0

    except KeyboardInterrupt:
        logger.warning("收到中断信号，停止worker")
        exit_code = 0

    except Exception as e:
        logger.error(f"worker运行失败: {e}")
        exit_code = 1

    finally:
        stop() # 停止worker   
        logger.info("===========worker运行结束===========\n\n")
        sys.exit(exit_code)