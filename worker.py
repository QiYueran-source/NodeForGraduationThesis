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
    parser.add_argument('--train_config', required=True, help='JSON 对象字符串')

    args = parser.parse_args()

    stock_list = json.loads(args.stock_list)
    earliest = json.loads(args.earliest_year_month)
    if isinstance(earliest, list):
        earliest = tuple(earliest)
    train_config = json.loads(args.train_config)

    # 写入meta缓存池（end_month 在 put_end_year 内固定为 12）
    DATA_CACHE_POOL.put_task_id(args.task_id)
    DATA_CACHE_POOL.put_start_year(args.start_year)
    DATA_CACHE_POOL.put_end_year(args.end_year)
    DATA_CACHE_POOL.put_N(args.N)
    DATA_CACHE_POOL.put_stock_list(stock_list)
    DATA_CACHE_POOL.put_earliest_year_month(*earliest)
    DATA_CACHE_POOL.put_train_config(train_config)

    # 记录
    DATA_CACHE_POOL.put_running(True)
    DATA_CACHE_POOL.put_pid(os.getpid())

parse_args_and_load_pool()

# 其他组件
from src.worker.data import data_thread  # 数据线程
from src.worker.save import SAVER  # 保存器
from src.worker.save import record_thread  # record 线程（写 record.json 供 tcp 查询）
from src.worker.agent import AGENT_DATA_ADAPTER, REWARD_MANAGER  # 数据适配器，奖励管理器
from src.worker.agent import MLP
from src.worker.agent import RollingEnv

def save_meta():
    """保存 meta 信息"""
    SAVER.save_meta()
    logger.debug("meta 信息保存完成")

def start_running():
    """设置运行状态"""  
    DATA_CACHE_POOL.put_running(True)

def start_datathread():
    """启动数据线程"""
    data_thread.data_thread_start()
    while not DATA_CACHE_POOL.contains_train(DATA_CACHE_POOL.get_start_year()):
         time.sleep(1)  # 等待1s保证数据加载完成
    logger.debug("数据线程启动完成")

def start_recordthread():
    """启动 record 线程（定时写 record.json 供 tcp 查询状态）"""
    record_thread.record_thread_start()
    logger.debug("record 线程启动完成")  

def train():
    """最简单训练循环：RollingEnv + MLP，每步 loss=-reward*log_prob 做 policy gradient 式更新。"""
    tc = DATA_CACHE_POOL.get_train_config() or {}
    n = int(tc.get("n", 1))
    m = int(tc.get("m", 1))
    mask_len = int(tc.get("mask_len", 60))

    env = RollingEnv()
    model = MLP(n, m, mask_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs, info = env.reset() # 数据加载器限制，一个task只能跑1轮  
    episode_steps = 0
    while DATA_CACHE_POOL.get_running():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # (1, n, m, mask_len)
        action_t = model(obs_t)  # (1, n+1)
        log_prob = (torch.log(action_t.clamp(1e-8)) * action_t).sum(dim=-1).squeeze(0)
        action_np = action_t.squeeze(0).detach().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        loss = -float(reward) * log_prob
        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode_steps += 1
        obs = next_obs
        if terminated or truncated:
            logger.debug("episode 结束 steps=%s reward=%s", episode_steps, reward)
            break
        if not DATA_CACHE_POOL.get_running():
            logger.info("训练循环退出")
            break
    
    

def stop():
    # 保存
    SAVER.append_performance_and_reward_snapshot()  # 最后一次落盘
    SAVER.save_model()  # 保存模型
    SAVER.save_status()  # 保存状态（写 record.json）

    # 停止
    record_thread.record_thread_stop()
    data_thread.data_thread_stop()

    # 设置运行状态
    DATA_CACHE_POOL.put_running(False)

    logger.debug("数据线程与 record 线程停止完成")


if __name__ == "__main__":
    try:
        parse_args_and_load_pool()
    except Exception as e:
        logger.error("解析参数失败: %s", e)
        sys.exit(1)

    try:
        logger.info("===========worker启动===========")
        logger.info("task_id: %s", DATA_CACHE_POOL.get_task_id())
        logger.info("start_year: %s", DATA_CACHE_POOL.get_start_year())
        logger.info("N: %s", DATA_CACHE_POOL.get_N())
        logger.info("stock_list: %s", DATA_CACHE_POOL.get_stock_list())
        logger.info("earliest_year_month: %s", DATA_CACHE_POOL.get_earliest_year_month())
        logger.info("train_config: %s", DATA_CACHE_POOL.get_train_config())

        start_running()  # 设置运行状态
        start_datathread()  # 启动数据线程
        start_recordthread()  # 启动 record 线程（写 record.json 供 tcp 查询）
        save_meta()  # 保存 meta 信息
        train()  # 开始训练

        stop()
        logger.info("===========worker运行结束===========\n\n")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("收到中断信号，停止worker")
        stop()
        logger.info("===========worker停止===========\n\n")
        sys.exit(0)

    except Exception as e:
        logger.error("worker运行失败: %s", e)
        stop()
        sys.exit(1)
