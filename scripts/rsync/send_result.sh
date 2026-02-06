#!/bin/bash

# 发送结果到主机
task_id="${task_id:?task_id 未设置}"
node_id="${node_id:?node_id 未设置}"
rsync -avz \
    --password-file=/rsync.passwd \
    --port=8730 /Node/data/$task_id \
    frank@43.139.192.176::results/$task_id/$node_id