#!/bin/bash

# 发送结果到主机（对应主机 rsyncd 模块 [node_result]，port 873，auth user nodeuser）
task_id="${task_id:?task_id 未设置}"
node_id="${node_id:?node_id 未设置}"

# 默认使用 frp 公网 IP/端口；若提供 RSYNC_HOST/RSYNC_PORT 环境变量，则优先使用内网配置。
default_frp_ip="43.139.192.176"
default_frp_port="8730"
frp_server_ip="${RSYNC_HOST:-$default_frp_ip}"
frp_server_port="${RSYNC_PORT:-$default_frp_port}"
# 只发送 meta、p&r、record，不发送 model.safetensors / checkpoint_rl.zip
rsync -avz \
    --progress \
    --info=progress2 \
    --password-file=/Node/rsync.passwd \
    --port="${frp_server_port}" \
    --include='record.json' \
    --include='meta.json' \
    --include='performance_and_reward_*.jsonl' \
    --exclude='*' \
    /Node/data/"$task_id"/ \
    "nodeuser@${frp_server_ip}::node_result/${task_id}/${node_id}"