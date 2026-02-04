#!/usr/bin/env python3
"""
TCP服务端
监听4321端口，接收TCP连接并打印收到的消息
参数通过命令行参数传给 worker，status 通过读取 worker 写入的 record.json 获取
"""
# 库
import subprocess
import signal
import socket
import sys
import json
import time
import os
from pathlib import Path

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, 'TCPReciver')

# tcp 进程内保存：当前 worker 的 pid 与 meta（便于获取 task_id、读 record.json）
_worker_pid = None
_meta = {}  # 含 task_id 等，用于 status 时拼 record.json 路径


def _is_process_alive(pid):
    """判断进程是否存活"""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_record_json(task_id):
    """从 /Node/data/{task_id}/record.json 读取状态，解析失败或文件不存在返回 None"""
    path = Path(f'/Node/data/{task_id}/record.json')
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.debug("读取 record.json 失败: %s", e)
        return None


def handle_message(message_str: str, socket_client: socket):
    """
    处理收到的消息
    消息格式：json {"req":1,"meta":{}}
    meta:
        - task_id: 任务id
        - start_year: 开始年份
        - end_year: 结束年份
        - end_month: 结束月份
        - N: 总股票数量
        - stock_list: 股票列表
        - earliest_year_month: 最早的年份和月份,(year, month) 或 [year, month]
        - train_config: 训练配置
    """
    global _worker_pid, _meta

    try:
        message = json.loads(message_str)
    except json.JSONDecodeError as e:
        logger.error("JSON解析失败: %s", e)
        return {"error": f"JSON解析失败: {e}"}

    req = int(message.pop('req'))

    if req == -1:
        # 停止 worker
        if _worker_pid and _is_process_alive(_worker_pid):
            try:
                os.kill(_worker_pid, signal.SIGTERM)
            except OSError as e:
                logger.warning("发送 SIGTERM 失败: %s", e)
        _worker_pid = None
        return {"stop": "success"}

    elif req == 0:
        # 查询状态：以 pid 判断是否在跑，从 record.json 读 task_id / current_year_month
        if _worker_pid and _is_process_alive(_worker_pid):
            task_id = _meta.get('task_id')
            current_year_month = None
            if task_id:
                data = _read_record_json(task_id)
                if data:
                    current_year_month = data.get('current_year_month')
            response = {
                "running": 1,
                "task_id": task_id,
                "current_year_month": current_year_month,
            }
        else:
            _worker_pid = None
            response = {"running": 0}
        return response

    elif req == 1:
        # 启动 worker：先检查是否已在跑，再写 run_config.json 并传 task_id
        if _worker_pid and _is_process_alive(_worker_pid):
            return {"error": "already running"}

        meta = message.pop('meta')
        task_id = meta.pop('task_id')

        data_dir = Path('/Node/data')
        task_path = data_dir / task_id
        if task_path.exists():
            return {"error": "task_id already exists"}

        task_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "worker.py",
            "--task_id", task_id,
            "--start_year", str(meta["start_year"]),
            "--end_year", str(meta["end_year"]),
            "--end_month", str(meta["end_month"]),
            "--N", str(meta["N"]),
            "--stock_list", json.dumps(meta["stock_list"], ensure_ascii=False),
            "--earliest_year_month", json.dumps(meta["earliest_year_month"], ensure_ascii=False),
            "--train_config", json.dumps(meta["train_config"], ensure_ascii=False),
        ]
        logger.info("启动 worker: python worker.py --task_id %s ...", task_id)
        try:
            proc = subprocess.Popen(cmd)
        except Exception as e:
            logger.error("启动 worker 失败: %s", e)
            return {"error": f"start failed: {e}"}

        _worker_pid = proc.pid
        _meta = {"task_id": task_id}

        time.sleep(5)
        if _is_process_alive(_worker_pid):
            response = {"success": "start success"}
        else:
            response = {"error": "start failed"}
        return response

    else:
        logger.error("req 取值错误: %s", req)
        return {"error": f"req字段取值错误: {req}"}


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = '127.0.0.1'
    port = 4321

    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        logger.info("TCP 服务器启动，监听 %s:%s", host, port)
        logger.info("等待连接和消息...")

        while True:
            client_socket, client_address = server_socket.accept()
            logger.info("收到来自 %s 的连接", client_address)
            try:
                data = client_socket.recv(1024)
                if data:
                    message_str = data.decode('utf-8', errors='strict').strip()
                    response = handle_message(message_str, client_socket)
                    logger.info("返回响应: %s", response)
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
            except Exception as e:
                logger.error("处理连接时出错: %s", e)
            finally:
                client_socket.close()
                logger.info("连接已关闭\n")

    except KeyboardInterrupt:
        logger.info("服务器关闭")
    except Exception as e:
        logger.error("服务器错误: %s", e)
        sys.exit(1)
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()
