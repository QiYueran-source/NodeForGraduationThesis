"""
测试奖励计算链路：测试期 202307~202312，rolling_window=6。
为 202307~202312 提供模拟收益率；仅对 202312 计算 vol/sharpe/回撤，并测试标准化。
"""
import json
import random
from pathlib import Path

# 必须先设置 DATA_CACHE_POOL，再导入会创建 AGENT_DATA_ADAPTER / REWARD_MANAGER 的模块
from src.worker.cache.pool import DATA_CACHE_POOL

# 初始化缓存池：task_id、当前窗口、适配器与 RewardManager 所需 meta（结构见 pool.py 顶部：顶层固定 + train_config 随机）
DATA_CACHE_POOL.put_task_id("test_reward_001")
DATA_CACHE_POOL.put_current_year_month(2023, 12)  # 测试当前窗口：202312
DATA_CACHE_POOL.put_start_year(2020)
DATA_CACHE_POOL.put_N(5)
DATA_CACHE_POOL.put_stock_list(["000001", "000002", "000003", "000004", "000005"])
DATA_CACHE_POOL.put_train_config({
    "n": 2,
    "max_portfolios_num": 10,
    "m": 1,
    "reward_config": {"reward_weights": {"rtr": 0.25, "vol": 0.25, "sharpe": 0.25, "max_drawdown": 0.25}},
})
DATA_CACHE_POOL.put_earliest_year_month(2020, 1)
DATA_CACHE_POOL.put_performance_config({
    "risk_free_rate": 0.02,
    "rolling_window": 6,  # 6 期滚动，仅对 202312 计算 vol/sharpe/max_drawdown
})

# 再导入 RewardManager、适配器、Saver
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER
from src.worker.save.saver import Saver


# 测试期：202307 ~ 202312
TEST_START_YEAR, TEST_START_MONTH = 2023, 7
TEST_END_YEAR, TEST_END_MONTH = 2023, 12
ROLLING_WINDOW = 6


def _roll_year_month(year: int, month: int, delta: int):
    """滚动年月，与 adapter 逻辑一致。"""
    total = year * 12 + (month - 1) + delta
    y = total // 12
    m = total % 12 + 1
    return y, m


def _inject_adapter_returns(year: int, month: int, code_to_rtr: dict):
    """在适配器 _train_data_pool 中注入模拟 (factors, rtr)，供 win_get_rtr 使用。"""
    for code, rtr in code_to_rtr.items():
        key = (year, month, code)
        AGENT_DATA_ADAPTER._train_data_pool[key] = ([], rtr)


def _inject_returns_202307_to_202311(portfolios: list, seed: int = 42):
    """
    为 202307~202311 仅注入模拟收益率到 _record（不计算 vol/sharpe/回撤）。
    供 202312 的 calc_portfolio_performance 滚动取 6 期 rtr。
    """
    rng = random.Random(seed)
    # 202307, 202308, 202309, 202310, 202311 共 5 期
    for i in range(1, ROLLING_WINDOW):  # 1..5 -> 往前 5 期
        y, m = _roll_year_month(TEST_END_YEAR, TEST_END_MONTH, -i)
        for p in portfolios:
            rtr = round(0.01 + (rng.random() - 0.5) * 0.04, 2)  # 约 -1% ~ 3%，保留两位小数
            key = (y, m, p)
            REWARD_MANAGER._record[key] = {
                "performance": [rtr],  # 仅收益率，其他指标不填
                "decision_weights": (0.5, 0.5, 0.0),
            }


def test_reward_calculation_flow():
    year, month = TEST_END_YEAR, TEST_END_MONTH  # 2023, 12
    AGENT_DATA_ADAPTER._current_year_month = (year, month)
    DATA_CACHE_POOL.put_current_year_month(year, month)

    # 3 个模拟组合（2 只标的 + 现金）
    portfolios = [
        ("000001", "000002"),
        ("000001", "000003"),
        ("000002", "000003"),
    ]

    # 1) 202307~202311：仅注入模拟收益率（其他指标不需要）
    _inject_returns_202307_to_202311(portfolios)

    # 2) 202312：注入当月标的收益率，供加权 rtr 计算
    _inject_adapter_returns(year, month, {"000001": 0.01, "000002": 0.02, "000003": -0.01})

    decision_weights_list = [
        (0.5, 0.5, 0.0),
        (0.6, 0.4, 0.0),
        (0.4, 0.6, 0.0),
    ]

    # 3) 仅计算 202312 的 performance（rtr + vol + sharpe + max_drawdown，用 6 期滚动）
    REWARD_MANAGER.calc_all_performance(year, month, portfolios, decision_weights_list)

    for p in portfolios:
        key = (year, month, p)
        rec = REWARD_MANAGER._record.get(key, {})
        assert "performance" in rec, f"缺少 performance: {key}"
        perf = rec["performance"]
        assert len(perf) == 4, f"performance 长度应为 4: {key}"
        rtr, vol, sharpe, max_dd = perf[0], perf[1], perf[2], perf[3]
        assert vol >= 0, f"vol 应为非负: {key}"
        assert isinstance(sharpe, (int, float)) and isinstance(max_dd, (int, float)), f"sharpe/max_drawdown 应为数值: {key}"

    # 4) 测试标准化
    REWARD_MANAGER.normalized_all_performance(year, month, portfolios)

    for p in portfolios:
        key = (year, month, p)
        rec = REWARD_MANAGER._record.get(key, {})
        assert "normalized_performance" in rec, f"缺少 normalized_performance: {key}"
        npf = rec["normalized_performance"]
        assert len(npf) == 4, f"normalized_performance 长度应为 4: {key}"
        # 标准化后应为数值
        for i, v in enumerate(npf):
            assert isinstance(v, (int, float)), f"normalized_performance[{i}] 应为数值: {key}"

    # 5) 计算奖励并保存（可选，保持与原测试一致）
    REWARD_MANAGER.calc_all_reward(year, month, portfolios)
    for p in portfolios:
        key = (year, month, p)
        rec = REWARD_MANAGER._record.get(key, {})
        assert "reward" in rec and isinstance(rec["reward"], (int, float)), f"reward 应为数值: {key}"

    # 6) 保存快照并校验
    REWARD_MANAGER._snapshot_progress = (0, 0)
    saver = Saver()
    record_path = saver.base_path / "record.jsonl"
    if record_path.exists():
        record_path.unlink()
    saver.save_performance_and_reward_snapshot()

    assert record_path.exists(), f"未生成文件: {record_path}"
    lines = [ln for ln in record_path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    assert len(lines) >= 3, f"至少应有 3 条记录（当前窗口 202312），实际 {len(lines)} 条"
    current_count = 0
    for ln in lines:
        row = json.loads(ln)
        if row["year"] == year and row["month"] == month:
            current_count += 1
            data = row["data"]
            assert "performance" in data and len(data["performance"]) == 4
            assert "normalized_performance" in data and "reward" in data
    assert current_count == 3, f"当前窗口 202312 应有 3 条，实际 {current_count} 条"

    print("test_reward_calculation_flow 通过")
    print(f"  测试期: {TEST_START_YEAR}{TEST_START_MONTH:02d}~{TEST_END_YEAR}{TEST_END_MONTH:02d}, rolling_window={ROLLING_WINDOW}")
    print(f"  仅对 {year}{month:02d} 计算 vol/sharpe/回撤并完成标准化")
    print(f"  记录: {record_path}, 总条数={len(lines)}, 当前窗口条数=3")


if __name__ == "__main__":
    test_reward_calculation_flow()
    import redis
    from src.worker.data.redis import REDIS_CONNECTOR
    client = REDIS_CONNECTOR.get_client()
    redis_client = redis.Redis()
    redis_client.get_retry()
    client.get()