#!/bin/bash

# 创建10个容器：rl_1 ～ rl_10
# 每个容器使用 node:gpu 镜像
# NODE_ID 与容器名相同
# FRP_PORT：8191 ～ 8200
IMAGE_NAME="rl-node"
IMAGE_TAG="gpu"
NODE_COUNT=5
START_INDEX=11
START_PORT=8201

echo "开始创建${NODE_COUNT}个容器..."

for i in $(seq $START_INDEX $((START_INDEX + NODE_COUNT - 1)))
do
    container_name="rl_${i}"
    frp_port=$((START_PORT + i - START_INDEX))

    echo "创建容器: ${container_name}, FRP_PORT: ${frp_port}"

    if docker ps -a --format '{{.Names}}' | grep -qx "${container_name}"; then
        echo "  已存在同名容器，先删除: ${container_name}"
        docker rm -f "${container_name}"
    fi

    docker run -d \
        --gpus all \
        --name "${container_name}" \
        -e NODE_ID="${container_name}" \
        -e FRP_PORT="${frp_port}" \
        --restart unless-stopped \
        $IMAGE_NAME:$IMAGE_TAG

    echo "容器 ${container_name} 创建完成"
done

echo "所有容器创建完成！"
echo ""
echo "验证容器状态："
docker ps --filter "name=rl_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"