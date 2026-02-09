"""
保存层  
负责调用保存接口获取快照，并保存到本地    
"""

# 库
import math
import yaml
import os
import json
import threading
from pathlib import Path

from safetensors.torch import save_file

# 自定义组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.env import REWARD_MANAGER
from src.worker.agent.net import NET_ADAPTER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[Saver]')

# 保存层
class Saver:
    def __init__(self):
        # 配置
        self._record_config = {} 

        # 加载配置 
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._record_config = yaml.safe_load(f).get('record_thread', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 

    def _get_base_path(self) -> Path:
        """按当前 task_id 返回基目录并确保存在"""
        task_id = DATA_CACHE_POOL.get_task_id()
        if not task_id:
            return Path('/Node/data')
        base = Path(f'/Node/data/{task_id}')
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _serialize_perf(self, data: dict) -> dict:
        """将表现 dict 转为 JSON 可序列化（tuple -> list）"""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in data.items()}

    def save_meta(self):
        """保存 meta 到本地（结构见 pool.py 顶部：顶层固定 + train_config 随机）"""
        meta_path = self._get_base_path() / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(DATA_CACHE_POOL.get_meta(), f, ensure_ascii=False)
        

    def _write_jsonl_worker(self, record_path: Path, lines: list):
        """后台线程：将已序列化的行追加写入 record.jsonl。"""
        try:
            with open(record_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            logger.debug(f"追加记录快照: {record_path}, 条数={len(lines)}")
        except Exception as e:
            logger.error(f"异步写入 record.jsonl 失败: {e}")

    def append_performance_and_reward_snapshot(self, daemon:bool = True):
        """异步保存表现和奖励快照到本地，按 JSONL 追加到 performance_and_reward.jsonl，不阻塞主流程。"""
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.warning("当前窗口未设置，跳过保存快照")
            return
        year, month = current_ym
        incremental_result = REWARD_MANAGER.get_incremental_snapshot(year, month)
        if not incremental_result:
            return
        record_path = self._get_base_path() / "performance_and_reward.jsonl"
        lines = []
        for key, data in incremental_result.items():
            y, m, portfolio = key
            obj = {"year": y, "month": m, "portfolio": list(portfolio), "data": self._serialize_perf(data)}
            obj = self._round_floats_in(obj, 3)  # 写入 performance_and_reward.jsonl 前统一 3 位有效数字
            lines.append(json.dumps(obj, ensure_ascii=False))
        threading.Thread(target=self._write_jsonl_worker, args=(record_path, lines), daemon=daemon).start() # 不守护，保证落盘陈功

    def _write_model_worker(self, state_dict: dict, out_path: str):
        """后台线程：将 state_dict 写入 safetensors 文件。"""
        try:
            save_file(state_dict, out_path)
            logger.info(f"模型已保存: {out_path}")
        except Exception as e:
            logger.error(f"异步保存模型失败: {e}")

    def save_model(self, daemon:bool = True):
        """异步保存模型到本地（safetensors），主线程仅做 get_checkpoint，写盘在后台执行，不阻塞。"""
        base = self._get_base_path()
        out_path = str(base / "model.safetensors")
        state_dict = NET_ADAPTER.get_checkpoint()
        threading.Thread(target=self._write_model_worker, args=(state_dict, out_path), daemon=daemon).start() 

    def _write_record_worker(self, base: Path, payload: dict):
        """后台线程：将 payload 原子写入 record.json。"""
        try:
            tmp_path = base / "record.json.tmp"
            record_path = base / "record.json"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, record_path)
            logger.debug(f"record.json 已写入: {record_path}")
        except Exception as e:
            logger.error(f"异步写入 record.json 失败: {e}")

    def save_record(self, daemon:bool = True):
        """异步将 task_id、current_year_month、record 写入 record.json（原子写），供 tcp 查询状态，不阻塞。"""
        task_id = DATA_CACHE_POOL.get_task_id()
        if not task_id:
            logger.debug("task_id 未设置，跳过写入 record.json")
            return
        base = self._get_base_path()
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        payload = {
            "task_id": task_id,
            "current_year_month": list(current_ym) if current_ym else None,
            "record": DATA_CACHE_POOL.get_record(),
        }
        threading.Thread(target=self._write_record_worker, args=(base, payload), daemon=daemon).start()

    def save_status(self):
        """保存状态到本地（与 save_record 一致，供 worker 停止时调用）"""
        self.save_record()

SAVER = Saver()

