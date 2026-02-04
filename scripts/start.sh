#!/bin/bash

cleanup() {
    echo "=== 收到停止信号，先执行 stop.sh ==="
    /Node/scripts/stop.sh
    exit 0
}

trap cleanup SIGTERM SIGINT

# frp启动和检查
echo "=== 启动frp客户端 ==="
/Node/scripts/frp/frp_start.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] frp启动失败，停止启动其他服务"
    exit 1
fi

# 检查frp状态
echo "=== 检查frp状态 ==="
/Node/scripts/frp/frp_status.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] frp状态检查失败，停止启动其他服务"
    exit 1
fi

# tcp监听启动
echo "=== 启动tcp监听进程 ==="
python /Node/tcp_reciver.py &



wait