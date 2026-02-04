#!/bin/bash
# frp客户端停止脚本：从状态文件读取 FRP_PID 和 ALLOCATED_PORT，kill 进程并释放端口

STATE_FILE="/Node/frp/frpc.state"
LOG_FILE="/Node/logs/frpc.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

echo "=== 停止frp客户端 ==="

if [ ! -f "$STATE_FILE" ]; then
    log_warn "状态文件不存在，frp客户端未运行或未通过 frp_start 启动"
    exit 0
fi

# 读取状态文件
# shellcheck source=/dev/null
source "$STATE_FILE"

if [ -z "$FRP_PID" ]; then
    log_warn "状态文件中无 FRP_PID，跳过停止"
    rm -f "$STATE_FILE"
    exit 0
fi

# 检查进程是否仍在运行
if ! kill -0 "$FRP_PID" 2>/dev/null; then
    log_warn "进程 $FRP_PID 已不存在，仅清理状态并释放端口"
else
    log_info "正在停止 frp 客户端 (PID: $FRP_PID)"
    kill "$FRP_PID" 2>/dev/null
    sleep 2
    if kill -0 "$FRP_PID" 2>/dev/null; then
        log_warn "普通 kill 未退出，尝试强制停止"
        kill -9 "$FRP_PID" 2>/dev/null
        sleep 1
    fi
fi

# 释放已分配的端口
if [ -n "$ALLOCATED_PORT" ]; then
    if curl -s "http://43.139.192.176:8190/release?port=$ALLOCATED_PORT" >/dev/null 2>&1; then
        log_info "已释放端口: $ALLOCATED_PORT"
    else
        log_warn "释放端口 $ALLOCATED_PORT 请求失败（请确认服务端 release 接口）"
    fi
fi

# 删除状态文件
rm -f "$STATE_FILE"
log_info "frp客户端已停止，状态文件已清理"
