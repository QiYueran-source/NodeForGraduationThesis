#!/usr/bin/env python3
"""
TCP服务端  
监听4321端口，接收TCP连接并打印收到的消息
"""
# 库
import subprocess
import socket
import sys
import json


# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__,'TCPReciver')

def handle_message(message_str: str,socket_client:socket):
    """
    处理收到的消息  
    消息格式：json {"req":1,"meta":{}}
    """
    # 解析JSON
    try:
        # 解析message
        message:dict = json.loads(message_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")

    
    # 获取req
    req = int(message.pop('req'))
    if not req or int(req) not in [-1,0,1]:
        logger.error('命令中没有req字段')
        raise ValueError
    
    # req可以取值-1,0,1
    if req == -1:
        response = "stop"
        socket_client.sendall(response.encode('utf-8'))
    if req == 0:
        response = "status"
        socket_client.sendall(response.encode('utf-8'))
    if req == 1:
        # 检查是否启动
        cmd = [

        ]
        response = "status"
        socket_client.sendall(response.encode('utf-8'))
        
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
                    message = handle_message(message_str)
                
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
    main()