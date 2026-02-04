#!/bin/bash
# frp客户端状态检查脚本（通过 frp/frpc.state 查询）

STATE_FILE="/Node/frp/frpc.state"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "=== frp客户端状态检查 ==="

# 从状态文件读取并检查进程是否存活
if [ ! -f "$STATE_FILE" ]; then
    log_warn "状态文件不存在: $STATE_FILE"
    log_info "frp客户端未运行或未通过 frp_start 启动"
else
    # shellcheck source=/dev/null
    source "$STATE_FILE"
    echo -e "${BLUE}[状态]${NC} FRP_PID=$FRP_PID  ALLOCATED_PORT=$ALLOCATED_PORT  STATE_TIME=$STATE_TIME"

    if [ -n "$FRP_PID" ] && kill -0 "$FRP_PID" 2>/dev/null; then
        log_info "frp客户端运行中 (PID: $FRP_PID)"
    else
        log_warn "frp客户端进程已退出或不存在 (PID: ${FRP_PID:-无})"
    fi
fi

# 检查配置文件
if [ -f "/Node/frp/frpc.toml" ]; then
    echo -e "${BLUE}[配置]${NC} /Node/frp/frpc.toml 存在"
else
    log_error "frpc.toml 配置文件不存在"
fi

# 检查日志文件
LOG_FILE="/Node/logs/frpc.log"
if [ -f "$LOG_FILE" ]; then
    echo -e "${BLUE}[日志]${NC} frpc.log 存在"
    echo -e "${BLUE}[日志]${NC} 最后几行:"
    tail -5 "$LOG_FILE" | sed 's/^/    /'
else
    log_warn "frpc.log 日志文件不存在"
fi

# 检查frp可执行文件
if [ -f "/Node/frp/frpc" ]; then
    FRP_VERSION=$(/Node/frp/frpc --version 2>/dev/null | head -1)
    if [ $? -eq 0 ]; then
        echo -e "${BLUE}[版本]${NC} $FRP_VERSION"
    else
        echo -e "${BLUE}[文件]${NC} frpc 可执行文件存在"
    fi
else
    log_error "frpc 可执行文件不存在"
fi

# 显示使用帮助
echo ""
echo -e "${BLUE}[帮助]${NC} 管理命令:"
echo "  启动: ./scripts/frp_start.sh"
echo "  停止: ./scripts/frp_stop.sh"
echo "  状态: ./scripts/frp_status.sh"
