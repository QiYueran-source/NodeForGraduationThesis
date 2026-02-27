#!/bin/bash

# 日志与状态文件
LOG_FILE="/Node/logs/frpc.log"
STATE_FILE="/Node/frp/frpc.state"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"
}

CONF="/Node/frp/frpc.toml"

# 检查配置文件是否存在
if [ ! -f "$CONF" ]; then
    log_error "配置文件不存在: $CONF"
    exit 1
fi

# 检查是否已经在运行（只检查使用该配置文件的 frpc 进程）
FRP_PID=$(ps aux 2>/dev/null | grep -v grep | grep "frpc -c $CONF" | awk '{print $2}' | head -1)
if [ -n "$FRP_PID" ]; then
    log_warn "frp客户端已在运行 (PID: $FRP_PID)"
    exit 0
fi

# 从配置文件解析 nodeName 和端口
NODE_NAME=$(
    grep -E '^name[[:space:]]*=' "$CONF" 2>/dev/null | head -1 | awk -F= '{print $2}' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | sed 's/^"//; s/"$//' | sed "s/^'//; s/'$//"
)
ALLOCATED_PORT=$(
    grep -E '^remotePort[[:space:]]*=' "$CONF" 2>/dev/null | head -1 | awk -F= '{print $2}' | tr -d '[:space:]'
)

if [ -z "$NODE_NAME" ]; then
    log_warn "未能从配置中解析到 node name，将仍然启动 frpc，但 frpc.state 中 NODE_NAME 为空"
fi

if [ -z "$ALLOCATED_PORT" ]; then
    log_warn "未能从配置中解析到 remotePort，将仍然启动 frpc，但 frpc.state 中 ALLOCATED_PORT 为空"
fi

log_info "使用配置文件启动 frp (nodeName=${NODE_NAME:-<unknown>}, port=${ALLOCATED_PORT:-<unknown>})"

# 启动frp客户端（日志由frpc.toml配置处理）
/Node/frp/frpc -c "$CONF" &
FRP_PID=$!  # 获取新进程的PID

# 检查是否启动完成  
sleep 3
if ! ps -p "$FRP_PID" > /dev/null 2>&1; then
    log_error "frp进程启动失败 - 进程不存在"
    exit 1
fi

# 持久化 FRP_PID、ALLOCATED_PORT、NODE_NAME、STATE_TIME 到状态文件，供 frp_stop / frp_status 使用
{
    echo "FRP_PID=$FRP_PID"
    echo "ALLOCATED_PORT=$ALLOCATED_PORT"
    echo "NODE_NAME=$NODE_NAME"
    echo "STATE_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
} > "$STATE_FILE"
log_info "frp进程启动成功 (PID: $FRP_PID, nodeName: ${NODE_NAME:-<unknown>})，状态已写入 $STATE_FILE"
