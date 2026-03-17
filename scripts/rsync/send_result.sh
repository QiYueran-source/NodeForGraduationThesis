#!/bin/bash

# 发送结果到主机（对应主机 rsyncd 模块 [node_result]，port 873，auth user nodeuser）
task_id="${task_id:?task_id 未设置}"
node_id="${node_id:?node_id 未设置}"
# 只发送 meta、p&r、record，不发送 model.safetensors / checkpoint_rl.zip
rsync -avz \
    --progress \
    --info=progress2 \
    --password-file=/Node/rsync.passwd \
    --port=8730 \
    --include='record.json' \
    --include='meta.json' \
    --include='performance_and_reward_*.jsonl' \
    --exclude='*' \
    /Node/data/"$task_id"/ \
    nodeuser@192.168.1.7::node_result/"$task_id"/"$node_id"