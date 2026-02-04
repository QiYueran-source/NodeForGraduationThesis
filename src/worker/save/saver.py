"""
保存层  
负责调用保存接口获取快照，并保存到本地    
"""

# 库
import yaml
import os
import json
import threading
import time
from pathlib import Path

# 自定义组件
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.env import REWARD_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[Saver]')

# 保存层
class Saver:
    def __init__(self):
        # 配置
        self._record_config = {} 

        # record 线程：定时将状态写入 record.json 供 tcp 查询
        self._record_thread = None
        self._record_started = False

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
        """保存元数据到本地"""
        meta_path = self._get_base_path() / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(DATA_CACHE_POOL.get_meta(), f, ensure_ascii=False)

    def append_performance_and_reward_snapshot(self):
        """保存表现和奖励快照到本地，按 JSONL 追加到 record.jsonl，每行一条 (year, month, portfolio) 完整记录。"""
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.warning("当前窗口未设置，跳过保存快照")
            return
        year, month = current_ym

        incremental_result = REWARD_MANAGER.get_incremental_snapshot(year, month)

        if incremental_result:
            record_path = self._get_base_path() / "record.jsonl"
            with open(record_path, "a", encoding="utf-8") as f:
                for key, data in incremental_result.items():
                    y, m, portfolio = key
                    obj = {"year": y, "month": m, "portfolio": list(portfolio), "data": self._serialize_perf(data)}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            logger.debug("追加记录快照: %s, 条数=%d", record_path, len(incremental_result))

    def save_model(self):
        """保存模型到本地"""
        pass 

    def save_record(self):
        """将 task_id、current_year_month、record 写入 record.json（原子写），供 tcp 查询状态。"""
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
        tmp_path = base / "record.json.tmp"
        record_path = base / "record.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, record_path)

    def save_status(self):
        """保存状态到本地（与 save_record 一致，供 worker 停止时调用）"""
        self.save_record()

    def _record_loop(self):
        """record 线程主循环：按间隔定时写入 record.json"""
        logger.info("record 线程启动")
        try:
            while self._record_started:
                try:
                    time.sleep(self._record_interval)
                    if not self._record_started:
                        break
                    self.save_record()
                except Exception as e:
                    logger.error("record 线程写入异常: %s", e)
                    time.sleep(self._record_config.get('record_interval', 5))
        except KeyboardInterrupt:
            logger.info("record 线程收到中断")
        logger.info("record 线程结束")

    def start_record_thread(self):
        """启动 record 线程"""
        if self._record_started:
            logger.warning("record 线程已在运行")
            return
        if self._record_thread and self._record_thread.is_alive():
            logger.warning("record 线程仍在运行中")
            return
        try:
            self._record_started = True
            self._record_thread = threading.Thread(
                target=self._record_loop,
                name="RecordThread",
                daemon=True,
            )
            self._record_thread.start()
            logger.info("record 线程已启动")
        except Exception as e:
            self._record_started = False
            logger.error("启动 record 线程失败: %s", e)
            raise

    def stop_record_thread(self):
        """停止 record 线程"""
        if not self._record_started:
            logger.info("record 线程未启动")
            return
        logger.info("正在停止 record 线程...")
        try:
            self._record_started = False
            if self._record_thread and self._record_thread.is_alive():
                self._record_thread.join(timeout=10)
                if self._record_thread.is_alive():
                    logger.warning("record 线程未在规定时间内结束")
            self._record_thread = None
            logger.info("record 线程已停止")
        except Exception as e:
            logger.error("停止 record 线程时发生错误: %s", e)
            self._record_thread = None

SAVER = Saver()

