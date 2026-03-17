#!/bin/bash

# 仅负责：从环境变量 FRP_PORT、NODE_ID 读取端口与节点名，并写入 /Node/frp/frpc.toml。必须设置二者，无兜底。

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

CONF="/Node/frp/frpc.toml"

if [ ! -f "$CONF" ]; then
    log_error "配置文件不存在: $CONF"
    exit 1
fi

# 必须使用环境变量 FRP_PORT 和 NODE_ID
if [ -z "${FRP_PORT}" ] || [ -z "${NODE_ID}" ]; then
    log_error "必须设置环境变量 FRP_PORT 和 NODE_ID"
    exit 1
fi
ALLOCATED_PORT="$FRP_PORT"
NODE_NAME="$NODE_ID"
# 若环境变量 USE_NODE_UUID 为 true/1，则在 node_id 后追加 8 位 uuid（默认 false）
if [ "${USE_NODE_UUID}" = "true" ] || [ "${USE_NODE_UUID}" = "1" ]; then
    _uuid=$(cat /proc/sys/kernel/random/uuid 2>/dev/null | tr -d '-' | cut -c1-8)
    if [ -n "$_uuid" ]; then
        NODE_NAME="${NODE_NAME}_${_uuid}"
    fi
fi
log_info "使用环境变量 (NODE_ID=$NODE_NAME, FRP_PORT=$ALLOCATED_PORT)"

# 注入 nodeName 和端口到配置文件（支持模板 name = $nodeName 或已有值覆盖）
if ! sed -i "s/^name = .*/name = \"$NODE_NAME\"/" "$CONF"; then
    log_error "nodeName 注入失败"
    exit 1
fi

if ! sed -i "s/^remotePort = .*/remotePort = $ALLOCATED_PORT/" "$CONF"; then
    log_error "端口注入失败"
    exit 1
fi

log_info "配置注入成功 (nodeName=$NODE_NAME, port=$ALLOCATED_PORT)"

