#!/usr/bin/env python3
"""
TCP服务端  
监听4321端口，接收TCP连接并打印收到的消息
"""
# 库
import subprocess
import signal
import socket
import sys
import json
import time 
import os

# 组件
from src.worker import DATA_CACHE_POOL


# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__,'TCPReciver')

def handle_message(message_str: str,socket_client:socket):
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
        # - factors_list: 因子列表（避免麻烦，直接保存本地）   
        - earliest_year_month: 最早的年份和月份,(year, month)  
        - train_config: 训练配置   
            - seed: 随机种子  
            - n: 一个组合中的证券数量（算上现金，共n+1个证券）  
            - max_portfolios_num: 对于总共n个证券，最多可以构建C(N,n)个组合,太大，所以设置最大组合数量    
            - m: 回看的期数      
            - mask_len: 因子掩码长度，默认60    
            - model_config: 模型配置（cate: 0 表示 mlp1；config: 模型具体参数）
            - performance_config: # 表现计算配置  
                - risk_free_rate: 无风险利率   
                - rolling_window: 滚动窗口期数  
            - reward_config: 奖励配置   
                - reward_weights: 奖励权重
                    - rtr: 收益率权重   
                    - vol: 波动权重   
                    - sharpe: 夏普比率权重   
                    - max_drawdown: 最大回测权重    

    """
    # 解析JSON
    try:
        # 解析message
        message:dict = json.loads(message_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        response = {
            "error": f"JSON解析失败: {e}"
        }
        return response

    # 获取req
    req = int(message.pop('req'))

    # req可以取值-1,0,1
    if req == -1:
        pid = DATA_CACHE_POOL.get_pid()
        if pid:
            os.kill(pid, signal.SIGTERM)
        response = {"stop": "success"}
        DATA_CACHE_POOL.put_pid(None)
        return response

    elif req == 0:
        is_running = DATA_CACHE_POOL.get_running()
        if is_running:
            task_id = DATA_CACHE_POOL.get_task_id()
            current_year_month = DATA_CACHE_POOL.get_current_year_month()
            response = {
                "running": 1,
                "task_id": task_id,
                "current_year_month": current_year_month,
            }
        else:
            response = {
                "running": 0,
            }
        return response

    elif req == 1:
        # 检查是否启动
        is_running = DATA_CACHE_POOL.get_running()
        if is_running:
            response = {
                "error": "already running"
            }
        else:
            # 获取meta
            meta = message.pop('meta')

            # 获取参数
            task_id = meta.pop('task_id')
            if f'/Node/data/{task_id}' in os.listdir('/Node/data'):
                response = {
                    "error": "task_id already exists"
                }
            else:
                start_year = meta.pop('start_year')
                end_year = meta.pop('end_year')
                end_month = meta.pop('end_month')
                N = meta.pop('N')
                stock_list = meta.pop('stock_list')
                earliest_year_month = meta.pop('earliest_year_month')
                train_config = meta.pop('train_config')
                
                # 设置参数
                DATA_CACHE_POOL.put_task_id(task_id)
                DATA_CACHE_POOL.put_start_year(start_year)
                DATA_CACHE_POOL.put_end_year(end_year)
                DATA_CACHE_POOL.put_end_month(end_month)
                DATA_CACHE_POOL.put_N(N)
                DATA_CACHE_POOL.put_stock_list(stock_list)
                DATA_CACHE_POOL.put_earliest_year_month(earliest_year_month)
                DATA_CACHE_POOL.put_train_config(train_config)
                
                # 启动
                cmd = [
                    "python",
                    "worker.py",
                ]
                logger.info(f"启动worker: {cmd}")
                proc = subprocess.Popen(cmd)

                # 保存进程号
                pid = proc.pid  
                DATA_CACHE_POOL.put_pid(pid)

                # 等待5s，如果worker启动成功，则返回成功
                time.sleep(5)
                if DATA_CACHE_POOL.get_running():
                    response = {
                        "success": "start success"
                    }
                else:
                    response = {
                        "error": "start failed"
                    }
        return response
    else:
        logger.error(f"req字段取值错误: {req}")
        response = {
            "error": f"req字段取值错误: {req}"
        }
        return response

def main():
    # 创建TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 绑定到本地4321端口
    host = '127.0.0.1'
    port = 4321
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        logger.info(f"TCP服务器启动，监听 {host}:{port}")
        logger.info("等待连接和消息...")
        
        while True:
            # 接受连接
            client_socket, client_address = server_socket.accept()
            logger.info(f"收到来自 {client_address} 的连接")
            
            # 接收数据
            try:
                data = client_socket.recv(1024)
                if data:
                    message_str = data.decode('utf-8', errors='strict').strip()
                    response = handle_message(message_str, client_socket)
                    logger.info(f"返回响应: {response}")
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"处理连接时出错: {e}")
            finally:
                client_socket.close()
                logger.info("连接已关闭\n")
                
    except KeyboardInterrupt:
        logger.info("服务器关闭")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        sys.exit(1)
    finally:
        server_socket.close()

if __name__ == "__main__":
    # 设置DATA_CACHE_POOL的running为False
    DATA_CACHE_POOL.put_running(False)

    # 启动
    main()