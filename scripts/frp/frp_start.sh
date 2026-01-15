#!/bin/bash

# 日志
LOG_FILE="/Node/logs/frpc.log"

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

# 获取可用端口 
ALLOCATED_PORT=$(curl -s http://43.139.192.176:8190/register | jq -r '.allocated_port')


# 端口不可用，则退出
if [ -z "$ALLOCATED_PORT" ] || [ "$ALLOCATED_PORT" = "null" ]; then
    log_error "无法获取可用端口"
    exit 1
fi
log_info "获取可用端口: $ALLOCATED_PORT"

# 导出到环境变量，供后续 Python 使用
export ALLOCATED_PORT="$ALLOCATED_PORT"
log_info "已导出环境变量 ALLOCATED_PORT"

# 注入端口到配置文件
if sed -i "s/remotePort = \$remotePort/remotePort = $ALLOCATED_PORT/" /Node/frp/frpc.toml; then
    log_info "端口注入成功"
else
    log_error "端口注入失败"
    exit 1
fi

# 检查是否已经在运行（只检查我们配置的frp进程）
FRP_PID=$(ps aux 2>/dev/null | grep -v grep | grep "frpc -c frp/frpc.toml" | awk '{print $2}' | head -1)
if [ -n "$FRP_PID" ]; then
    log_warn "frp客户端已在运行"
    exit 0
fi

# 启动frp客户端（日志由frpc.toml配置处理）
/Node/frp/frpc -c /Node/frp/frpc.toml &
FRP_PID=$!  # 获取新进程的PID

# 检查是否启动完成  
sleep 3
if ! ps -p $FRP_PID > /dev/null 2>&1; then
    log_error "frp进程启动失败 - 进程不存在"
    exit 1
fi

log_info "frp进程启动成功 (PID: $FRP_PID)"



