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
import subprocess
from pathlib import Path
import torch

# 自定义组件
from src.worker.data.redis import REDIS_CONNECTOR,REDIS_PREFIX_MANAGER

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
    parser.add_argument('--train_config', required=True, help='JSON 对象字符串（仅随机部分，结构见 pool.py 顶部）')

    args = parser.parse_args()

    # 通过redis获取meta数据 
    client = REDIS_CONNECTOR.get_client()
    meta_key = REDIS_PREFIX_MANAGER.build_meta_key()
    meta_json = client.get(meta_key)
    if meta_json is None:
        logger.error(f"meta not found in redis")
        sys.exit(1)
    meta = json.loads(meta_json)
    logger.info(f"获取meta数据: {meta}")

    # 写入 meta 缓存池（结构见 pool.py 顶部：顶层固定 + train_config 随机；end_month 在 put_end_year 内固定为 12）
    DATA_CACHE_POOL.put_task_id(meta['task_id'])
    DATA_CACHE_POOL.put_start_year(meta['start_year'])
    DATA_CACHE_POOL.put_end_year(meta['end_year'])
    DATA_CACHE_POOL.put_N(meta['N'])
    DATA_CACHE_POOL.put_stock_list(meta['stock_list'])
    year, month = meta['earliest_year_month']
    DATA_CACHE_POOL.put_earliest_year_month(year, month)
    DATA_CACHE_POOL.put_n(meta['n'])
    DATA_CACHE_POOL.put_max_portfolios_num(meta['max_portfolios_num'])
    DATA_CACHE_POOL.put_performance_config(meta['performance_config'])
    DATA_CACHE_POOL.put_env_config(meta['env_config'])
    DATA_CACHE_POOL.put_short_limit(meta['short_limit'])
    checkpoint_enabled = meta.get('checkpoint', False)
    DATA_CACHE_POOL.put_checkpoint(checkpoint_enabled)
    
    logger.info(f"断点增量：meta.checkpoint={checkpoint_enabled}")
    train_config_json = args.train_config
    train_config = json.loads(train_config_json)
    DATA_CACHE_POOL.put_train_config(train_config)

try:
    parse_args_and_load_pool()
except Exception as e:
    logger.error(f"解析参数失败: {e}")
    sys.exit(1)


# 其他组件
from src.worker.data import data_thread  # 数据线程
from src.worker.save import SAVER  # 保存器
from src.worker.agent import AGENT_DATA_ADAPTER, NET_ADAPTER, RL_ADAPTER, ROLLING_ENV  # agent组件 

def read_node_id_from_frp_state():
    """从 frp 状态文件中获取节点id"""
    path = Path("/Node/frp/frpc.state")
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_NAME="):
                return line.split("=", 1)[1].strip()
    return None

def start():
    # 设置运行状态
    DATA_CACHE_POOL.put_running(True)
    DATA_CACHE_POOL.put_pid(os.getpid())
    DATA_CACHE_POOL.put_node_id(read_node_id_from_frp_state())

    # 初始保存meta、record数据
    SAVER.save_meta() # 保存meta数据   
    SAVER.save_record() # 保存record数据  

    # 启动数据线程 
    data_thread.data_thread_start()  # 启动数据线程 
    time.sleep(10)  # 等待10s保证数据加载完成  

    logger.info("数据线程启动完成，开始训练")


def train():
    """使用NET_ADAPTER,ENV和RL_ADAPTER进行训练"""
    logger.info("开始强化学习训练")
    # 断点：每次 if 过滤一种不可用情况，全部通过后再加载
    checkpoint_json_path = Path("/Node/checkpoint.json")
    checkpoint_rl_path = Path("/Node/checkpoint_rl")
    checkpoint_rl_zip_path = Path("/Node/checkpoint_rl.zip")
    checkpoint_safetensors_path = Path("/Node/checkpoint.safetensors")
    rl_exists = checkpoint_rl_path.exists() or checkpoint_rl_zip_path.exists()
    use_checkpoint = False
    try:
        if not DATA_CACHE_POOL.get_checkpoint():
            logger.info("断点增量：meta.checkpoint 未启用，从头训练")
        elif not checkpoint_json_path.exists():
            logger.info("断点：checkpoint.json 不存在，从头训练")
        else:
            with open(checkpoint_json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_uuid = saved.get("config_uuid")
            current_uuid = (DATA_CACHE_POOL.get_train_config() or {}).get("config_uuid")
            if saved_uuid is None or current_uuid is None or saved_uuid != current_uuid:
                logger.info(f"断点：config_uuid 不一致或缺失，从头训练 (saved={saved_uuid!r}, current={current_uuid!r})")
            elif not rl_exists and not checkpoint_safetensors_path.exists():
                logger.info("断点：config_uuid 一致但 checkpoint 文件不存在，从头训练")
            else:
                tc = DATA_CACHE_POOL.get_train_config() or {}
                current_n = DATA_CACHE_POOL.get_n()
                if (saved.get("seed") != tc.get("seed") or saved.get("n") != current_n
                        or saved.get("m") != tc.get("m") or saved.get("mask_len") != tc.get("mask_len")):
                    logger.info(
                        f"断点：seed/n/m/mask_len 与当前不一致，从头训练 "
                        f"(saved seed={saved.get('seed')},n={saved.get('n')},m={saved.get('m')},mask_len={saved.get('mask_len')}; "
                        f"current seed={tc.get('seed')},n={current_n},m={tc.get('m')},mask_len={tc.get('mask_len')})"
                    )
                else:
                    use_checkpoint = True
        if use_checkpoint:
            if rl_exists:
                rl_path_str = str(checkpoint_rl_zip_path if checkpoint_rl_zip_path.exists() else checkpoint_rl_path)
                RL_ADAPTER.load_checkpoint(rl_path_str)
                logger.info("已从断点加载 RL 完整状态，开始增量训练")
            else:
                NET_ADAPTER.load_checkpoint(str(checkpoint_safetensors_path))
                logger.info("已从断点加载模型(仅权重)，开始增量训练")
    except Exception as e:
        logger.exception(f"断点加载跳过: {e}")

    try:
        RL_ADAPTER.train()
        logger.info("强化学习训练结束，进入预测阶段")
    except Exception:
        logger.exception("强化学习训练失败，保存当前状态后继续滚动预测")
        SAVER.save_record()
        SAVER.append_performance_and_reward_snapshot()
        SAVER.save_model(daemon=False)
        SAVER.save_rl_checkpoint(daemon=False)

    # 使用模型继续预测，停止条件：get_running() 为 False 或 year > end_year
    logger.info("========= 开始滚动预测 =========")
    print("========= 开始滚动预测 =========")
    _save_cursor = 0
    while True:
        if not DATA_CACHE_POOL.get_running():
            break
        ym = DATA_CACHE_POOL.get_current_year_month()
        if ym is None:
            logger.warning("当前窗口未设置，结束预测")
            break
        year, month = ym
        if year > DATA_CACHE_POOL.get_end_year():
            logger.info(f"预测阶段结束(已到end_year), year={year}")
            SAVER.save_record()
            SAVER.append_performance_and_reward_snapshot(segment=True)
            break

        # 预测
        obs, info = ROLLING_ENV.reset()

        # 没有可用的portfolio
        if info.get("no_more_episodes"):
            logger.info("预测阶段结束(no_more_episodes)")
            SAVER.save_record()
            SAVER.append_performance_and_reward_snapshot(segment=True)
            break

        # 行动
        try:
            action, _ = RL_ADAPTER.algorithm.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = ROLLING_ENV.step(action)
        except Exception:
            logger.exception("行动失败，跳过当前组合")
            continue

        # 保存结果
        ec = DATA_CACHE_POOL.get_env_config() or {}
        save_record_every_n_steps = ec.get('save_record_every_n_steps', 50)
        _save_cursor += 1
        if _save_cursor % save_record_every_n_steps == 0 and _save_cursor >= save_record_every_n_steps:
            SAVER.save_record()
    logger.info('========= 滚动预测结束 =========')
    print('========= 滚动预测结束 =========')
            

def send_result():
    """发送结果到主机（调用 rsync 脚本，需 task_id 与 frpc.state 中的 NODE_NAME）。"""
    logger.info("开始发送结果到主机")
    try:
        task_id = DATA_CACHE_POOL.get_task_id()
        if not task_id:
            logger.warning("task_id 为空，跳过发送结果")
            return
        node_id = read_node_id_from_frp_state()
        if not node_id:
            logger.warning("未从 frpc.state 读取到 NODE_NAME，跳过发送结果")
            return
        env = {**os.environ, "task_id": task_id, "node_id": node_id}
        script_path = Path("/Node/scripts/rsync/send_result.sh")
        if not script_path.exists():
            logger.warning(f"send_result.sh 不存在，跳过发送结果: {script_path}")
            return
        subprocess.run(
            ["bash", str(script_path)],
            env=env,
            check=True,
            cwd="/Node",
        )
        logger.info(f"发送结果到主机完成 (task_id={task_id}, node_id={node_id})")
    except subprocess.CalledProcessError as e:
        logger.exception(f"rsync 脚本执行失败 (exit {e.returncode})")
        sys.exit(1)
    except Exception:
        logger.exception("发送结果到主机失败")
        sys.exit(1)
        
        
def stop():
    # 设置运行状态
    DATA_CACHE_POOL.put_running(False)

    # 保存（仅写 data/task_id，不写 /Node）
    SAVER.append_performance_and_reward_snapshot(segment=True, daemon=False)  # 最后一次落盘，新段避免覆盖主机端 p&r_2
    t_model = SAVER.save_model(daemon=False)  # 保存模型到任务目录
    t_rl = SAVER.save_rl_checkpoint(daemon=False)  # 保存 RL 到任务目录
    SAVER.save_record(daemon=False)  # 保存状态（异步写 record.json）

    # 等待 model / RL 保存线程完成后再复制到 /Node，避免复制到半成品
    for t in (t_model, t_rl):
        if t is not None:
            t.join(timeout=30)
    time.sleep(0.5)  # 给 record 等异步写留一点时间

    # 发送前：先写 checkpoint.json，再复制 model/rl 到 /Node，保证 metadata 与文件一致
    SAVER.write_checkpoint_json()
    SAVER.copy_checkpoint_to_node()
    
    # 最后再发一次 perf 与 record，避免 daemon 写盘线程未及发送
    SAVER.send_perf_and_record()

    # 停止
    data_thread.data_thread_stop()

    logger.debug("数据线程与 record 线程停止完成")


if __name__ == "__main__":
    try:
        logger.info("===========worker启动===========")
        logger.info(f"task_id: {DATA_CACHE_POOL.get_task_id()}")
        logger.info(f"start_year: {DATA_CACHE_POOL.get_start_year()}")
        logger.info(f"N: {DATA_CACHE_POOL.get_N()}")
        logger.info(f"stock_list: {DATA_CACHE_POOL.get_stock_list()}")
        logger.info(f"earliest_year_month: {DATA_CACHE_POOL.get_earliest_year_month()}")
        logger.info(f"train_config: {DATA_CACHE_POOL.get_train_config()}")

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
        print('====== worker运行失败 ======')
        print(e)
        logger.error(f"worker运行失败: {e}")
        exit_code = 1

    finally:
        stop()  # 停止 worker
        send_result()  # 推送结果到主机
        logger.info("===========worker运行结束===========\n\n")
        sys.exit(exit_code)