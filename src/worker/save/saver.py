"""
保存层  
负责调用保存接口获取快照，并保存到本地    
"""

# 库
import os
import json 
from pathlib import Path

# 自定义组件 
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.reward import REWARD_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[Saver]')

# 保存层
class Saver:
    def __init__(self):
        # 基目录
        self.base_path = Path(f'/Node/data/{DATA_CACHE_POOL.get_task_id()}')

        # 创建基目录
        self._make_base_dir()

    def _make_base_dir(self):
        """创建基目录"""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _serialize_perf(self, data: dict) -> dict:
        """将表现 dict 转为 JSON 可序列化（tuple -> list）"""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in data.items()}

    def save_performance_and_reward_snapshot(self):
        """保存表现和奖励快照到本地，按 JSONL 追加到 record.jsonl，每行一条 (year, month, portfolio) 完整记录。"""
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.warning("当前窗口未设置，跳过保存快照")
            return
        year, month = current_ym

        incremental_result = REWARD_MANAGER.get_incremental_snapshot(year, month)

        if incremental_result:
            record_path = self.base_path / "record.jsonl"
            with open(record_path, "a", encoding="utf-8") as f:
                for key, data in incremental_result.items():
                    y, m, portfolio = key
                    obj = {"year": y, "month": m, "portfolio": list(portfolio), "data": self._serialize_perf(data)}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            logger.debug("追加记录快照: %s, 条数=%d", record_path, len(incremental_result))

SAVER = Saver()

