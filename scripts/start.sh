#!/bin/bash

# 启动脚本

# frp启动和检查  
echo "=== 启动frp客户端 ==="
/Node/scripts/frp/frp_start.sh 
if [ $? -ne 0 ]; then
    echo "[ERROR] frp启动失败，停止启动其他服务"
    exit 1
fi

# worker启动  
echo "=== 启动worker进程 ==="
python /Node/worker.py &

echo "=== 启动测试TCP服务器(开发模式) ==="
python /Node/test_tcp_server.py &

echo "=== 所有服务已启动 ==="

# 打印frp状态
echo "=== 查看frp状态(开发环境) ==="
/Node/scripts/frp/frp_status.sh

# 等待所有后台进程
wait