# TODO：主机端生成 meta 需添加的字段

节点端已支持以下能力，主机在生成并下发的 **meta** 中需提供对应字段，否则使用默认行为。

## 顶层（固定）

| 路径 | 类型 | 说明 | 默认/必填 |
|------|------|------|-----------|
| `env_config.retrain_times` | int | 同一窗口重复训练轮数；>1 时组合用尽后先重置游标并打乱再扫一轮，满轮后再滚窗 | 默认 1，不提供则单轮 |
| `env_config.save_every_n_steps` | int | 每多少步保存一次模型 | 可选 |
| `env_config.sample_and_shuffle_seed` | int | 采样与滚窗打乱种子，可复现 | 可选 |
| `performance_config.classic_utility` | bool | `true` 时奖励用博迪效用 U=μ-(A/2)σ²；`false` 时用 reward_weights 加权归一化指标 | 不提供视为 false，走原奖励 |
| `performance_config.risk_free_rate` | float | 无风险利率（夏普等） | 可选，默认 0.02 |
| `performance_config.std_window` | int | 纵向标准化窗口期数 | 可选 |
| `performance_config.std_floor` | float | 标准化分母下限，防 z-score 爆炸 | 可选，默认 0.01 |

## train_config（随机，每个 agent 可不同）

| 路径 | 类型 | 说明 | 默认/必填 |
|------|------|------|-----------|
| `reward_config.A` | float | 博迪效用风险厌恶系数，仅当 `performance_config.classic_utility=true` 时生效；可随机初始化 | 不提供时节点用 2.0 |
| `reward_config.reward_weights` | dict | rtr/vol/sharpe/max_drawdown/diversification 权重，仅当 classic_utility=false 时生效 | 可选 |

## 小结（新增/易漏）

- **同一窗口多轮训练**：在 `env_config` 中提供 `retrain_times`（如 2、3）。
- **博迪效用奖励**：在 `performance_config` 中设 `classic_utility: true`，并在 `train_config.reward_config` 中提供 `A`（如随机正数）。

完整 meta 结构见 `src/worker/cache/pool.py` 文件顶部注释。
