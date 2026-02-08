#!/bin/bash

# 发送结果到主机（对应主机 rsyncd 模块 [node_result]，port 873，auth user nodeuser）
task_id="${task_id:?task_id 未设置}"
node_id="${node_id:?node_id 未设置}"
rsync -avz \
    --progress \
    --info=progress2 \
    --password-file=/Node/rsync.passwd \
    --port=8730 \
    /Node/data/"$task_id"/ \
    nodeuser@43.139.192.176::node_result/"$task_id"/"$node_id"