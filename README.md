# 节点

## 命令  
以json来启动  

字段:  
- 1.req 
    - 1:启动命令   
    - 0:状态查询命令  
    - -1：停止命令  
- 2.meta: (仅启动命令带相关内容)  
    - task_id: 任务id    
    - start_year: 开始年份  
    - end_year: 停止年份（结束月份固定为 12，不需提供 end_month）
    - N: 总股票数量    
    - stock_list: 股票列表   
    - factors_list: 因子列表（避免麻烦，直接保存本地）   
    - earliest_year_month: 最早的年份和月份,(year, month)  
    - train_config: 训练配置   
        - seed: 随机种子  
        - n: 一个组合中的证券数量（算上现金，共n+1个证券）  
        - max_portfolios_num: 对于总共n个证券，最多可以构建C(N,n)个组合,太大，所以设置最大组合数量    
        - m: 回看的期数      
        - mask_len: 因子掩码长度，默认60    
        - model_config: 模型配置
            - cate: 模型类别，0 表示 mlp1  
            - cuda: 是否使用cuda,1表示使用，0表示不使用  
            - opt: 
                - cate: 优化器类别，0表示adam，1表示sgd
                - lr: 学习率
                - weight_decay: L2正则化系数(如果优化器支持) 
            - clip_grad_norm: 梯度裁剪范数  
            - dropout:  dropout率  
            - config: 具体模型参数(不同模型不同参数)
        - performance_config: # 表现计算配置  
- risk_free_rate: 无风险利率
        - （vol/sharpe/max_drawdown 的滚动窗口与 train_config.m 一致，不再单独配置 rolling_window）
        - reward_config: 奖励配置   
            - reward_weights: 奖励权重
                - rtr: 收益率权重   
                - vol: 波动权重   
                - sharpe: 夏普比率权重   
                - max_drawdown: 最大回测权重    

## 回应

服务监听 **127.0.0.1:4321**，每次连接发送一条 JSON 请求，服务端返回一条 JSON 响应后关闭连接。请求与响应均为 UTF-8 编码的 JSON 字符串。

### 按 req 的响应说明

| req | 含义 | 可能响应 |
|-----|------|----------|
| 1 | 启动命令 | 见下方「启动 (req=1)」 |
| 0 | 状态查询 | 见下方「状态查询 (req=0)」 |
| -1 | 停止命令 | 见下方「停止 (req=-1)」 |
| 其他 | 非法 | `{"error": "req字段取值错误: {req}"}` |

### 启动 (req=1)

- **成功**：`{"success": "start success"}`  
  表示参数已写入缓存并已拉起 worker，约 5 秒内 worker 将 running 置为真后返回。
- **已有人在跑**：`{"error": "already running"}`  
  当前节点已在执行任务，不能再次启动。
- **task_id 已占用**：`{"error": "task_id already exists"}`  
  本机已存在目录 `/Node/data/{task_id}`，不允许重复使用该 task_id。
- **启动失败**：`{"error": "start failed"}`  
  worker 进程已拉起，但约 5 秒内未将 running 置为真（可能启动报错或异常退出）。

启动命令必须带 `meta` 字段，且 `meta` 内需包含：`task_id`, `start_year`, `end_year`, `N`, `stock_list`, `earliest_year_month`, `train_config`。缺字段会在服务端 pop 时抛错，未单独做字段级错误码。（end_month 固定为 12，主机不需提供。）

### 状态查询 (req=0)

- **运行中**：`{"running": 1, "task_id": "<任务id>", "current_year_month": [年, 月]}`  
  `current_year_month` 为当前训练/推理的时间窗口。
- **未运行**：`{"running": 0}`  

不需要带 `meta`。

### 停止 (req=-1)

- **成功**：`{"stop": "success"}`  
  已向当前 worker 进程发送 SIGTERM，并清空缓存的 pid。  
不需要带 `meta`。

### 通用错误

- **请求体非合法 JSON**：`{"error": "JSON解析失败: {具体异常信息}"}`  

### 示例

- 状态查询请求：`{"req": 0}`  
  响应示例：`{"running": 1, "task_id": "task_001", "current_year_month": [2020, 6]}` 或 `{"running": 0}`  
- 停止请求：`{"req": -1}`  
  响应：`{"stop": "success"}`  
- 启动请求：`{"req": 1, "meta": { "task_id": "task_001", "start_year": 2018, ... }}`  
  成功响应：`{"success": "start success"}`  
