#!/bin/bash

# 仅负责：向注册中心申请端口 + 生成 NODE_NAME，并将二者写入 /Node/frp/frpc.toml。

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

# 向注册中心申请端口
ALLOCATED_PORT=$(curl -s http://43.139.192.176:8190/register | jq -r '.allocated_port')

# 端口不可用，则退出
if [ -z "$ALLOCATED_PORT" ] || [ "$ALLOCATED_PORT" = "null" ]; then
    log_error "无法获取可用端口"
    exit 1
fi
log_info "获取可用端口: $ALLOCATED_PORT"

# 随机生成 nodeName（node_时间戳_随机数，保证唯一）
NODE_NAME="node_$(date +%s)_${RANDOM}"
log_info "生成 nodeName: $NODE_NAME"

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

