#!/bin/bash

# 停止frpc进程，释放端口
echo "=== 停止frpc进程，释放端口 ==="
/Node/scripts/frp/frp_stop.sh

# 停止tcp监听进程
echo "=== 停止tcp监听进程 ==="
pkill -f "/Node/tcp_reciver.py"