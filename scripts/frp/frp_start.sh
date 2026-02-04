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

# 获取可用端口 
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
if ! sed -i "s/^name = .*/name = \"$NODE_NAME\"/" /Node/frp/frpc.toml; then
    log_error "nodeName 注入失败"
    exit 1
fi
if ! sed -i "s/^remotePort = .*/remotePort = $ALLOCATED_PORT/" /Node/frp/frpc.toml; then
    log_error "端口注入失败"
    exit 1
fi
log_info "配置注入成功 (nodeName=$NODE_NAME, port=$ALLOCATED_PORT)"

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

# 持久化 FRP_PID、ALLOCATED_PORT、NODE_NAME、STATE_TIME 到状态文件，供 frp_stop / frp_status 使用
echo "FRP_PID=$FRP_PID" > "$STATE_FILE"
echo "ALLOCATED_PORT=$ALLOCATED_PORT" >> "$STATE_FILE"
echo "NODE_NAME=$NODE_NAME" >> "$STATE_FILE"
echo "STATE_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> "$STATE_FILE"
log_info "frp进程启动成功 (PID: $FRP_PID, nodeName: $NODE_NAME)，状态已写入 $STATE_FILE"



