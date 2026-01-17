#!/bin/bash

# 启动脚本

# frp启动和检查  
echo "=== 启动frp客户端 ==="
/Node/scripts/frp/frp_start.sh 
if [ $? -ne 0 ]; then
    echo "[ERROR] frp启动失败，停止启动其他服务"
    exit 1
fi

# tcp监听启动  
echo "=== 启动tcp监听进程 ==="
python /Node/tcp_listener.py & 

wait