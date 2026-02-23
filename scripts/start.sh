#!/bin/bash

export TERM=dumb

cleanup() {
    echo "=== 收到停止信号，先执行 end.sh ==="
    /Node/scripts/end.sh
    exit 0
}

trap cleanup SIGTERM SIGINT

# tcp监听启动
# 先启动tcp监听4321，才能启动frp  
echo "=== 启动tcp监听进程 ==="
python /Node/tcp_reciver.py &

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

wait