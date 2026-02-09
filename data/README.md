# 节点产出数据说明

本目录存放节点（Worker）在每次任务下产生的结果数据，按 `task_id` 分子目录。

---

## 数据来源

- **任务元信息（meta）**：由主机在启动任务时下发，包括 `task_id`、`start_year`、`end_year`、`N`、`stock_list`、`earliest_year_month`、`n`、`max_portfolios_num`、`performance_config`、`env_config` 以及 `train_config`（随机部分）等，写入 `meta.json`。
- **训练与因子数据**：由主机提供并写入数据缓存池，键为 `(code, year, month)`，值为因子与收益等；节点在滚动时间窗口内按组合（portfolio）取因子与收益率，用于环境观测与表现计算。
- **表现与奖励**：在节点内由 `RewardManager` 根据当期的组合权重（agent 决策）、历史收益率序列与配置，在线计算并写入 `performance_and_reward.jsonl`。

---

## 奖励计算方法

- **表现指标（performance）**：对每个 `(year, month, portfolio)` 计算 4 项（顺序固定）  
  - 收益率 **rtr**：当期加权收益率 `sum(decision_weights * 收益率)`，含现金。  
  - 波动率 **vol**：使用 `performance_config.rolling_window` 期收益率序列的 `std`，保存时取**负数**（波动越大惩罚越大）。  
  - 夏普比率 **sharpe**：`(收益率序列均值 - risk_free_rate) / (std + 1e-6)`。  
  - 最大回撤 **max_drawdown**：基于同一条收益率序列的累计净值计算，保存时取**负数**。  

- **纵向标准化（normalized_performance）**：对上述 4 项分别做 z-score。  
  - 使用 `performance_config.std_window` 期，从当前 `(year, month)` **往前**取过去若干期；  
  - 在这些期内收集**所有组合**的对应指标值，计算均值和标准差（std 加 1e-6 防除零）；  
  - 当前组合的该项指标做 `(x - mean) / std`，得到 4 维 `normalized_performance`。  

- **奖励（reward）**：  
  - `reward = sum(reward_weights[k] * normalized_performance[k])`，其中 `k` 依次为 `rtr`、`vol`、`sharpe`、`max_drawdown`。  
  - `reward_weights` 来自 `train_config.reward_config.reward_weights`，在节点内会归一化为和 1。

---

## NaN 填充方式

- **标准化结果**：在 `normalize_portfolio_performance` 中，对计算得到的 `normalized_performance` 逐项检查；若为 NaN 或 Inf（`np.isfinite` 为 False），则**用 0 填充**后再写入内存记录并参与后续 reward 计算，避免 NaN 沿纵向窗口传播到后续期。  
- **奖励计算**：在 `calc_portfolio_reward` 中，对 `normalized_performance` 的每一项在加权前做兜底：若该项非有限数，则按 0 参与加权求和，保证 `reward` 不为 NaN。

---

## 目录与文件说明

每个任务对应子目录：`data/<task_id>/`。

| 文件 | 说明 |
|------|------|
| `meta.json` | 任务元信息（顶层固定 + train_config 随机部分），与主机下发的 meta 一致。 |
| `record.json` | 运行状态：`task_id`、`current_year_month`、以及节点维护的 `record`（如 running、pid、node_id 等），供 TCP 状态查询。 |
| `performance_and_reward.jsonl` | 表现与奖励快照，每行一条 JSON：`year`、`month`、`portfolio`、`data`。`data` 含 `decision_weights`、`performance`（4 维）、`normalized_performance`（4 维）、`reward`。浮点数在写入前**统一保留 3 位有效数字**。 |
| `model.safetensors` | 当前任务训练得到的模型参数（如 PPO policy 等），按需保存。 |

---

## 其他说明

- 若某期获取历史收益率序列时存在缺失，会汇总为一条警告日志（“获取收益率序列失败: 共 N 期 …”），不会对每个缺失期单独打一条。  
- 纵向标准化时，若过去 `std_window` 期内某指标样本数不足 2，则对该指标使用 `mean=0, std=1`，不做 z-score 除法。
