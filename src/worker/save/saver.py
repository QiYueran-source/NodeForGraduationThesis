"""
保存层  
负责调用保存接口获取快照，并保存到本地    
"""

# 库
import math
import yaml
import os
import json
import subprocess
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

        # 文件游标与锁（保证 segment 与文件名一致，多线程安全）
        self._performance_and_reward_snapshot_cursor = 0
        self._snapshot_lock = threading.Lock()
        self._send_lock = threading.Lock()  # 落盘+发送：只发 record + perf，发送成功后只删本次列表中的 perf

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

    def _round_floats_in(self, obj, ndigits: int):
        """递归将 dict/list 中的 float 统一保留 ndigits 位小数，便于 JSON 落盘可读。"""
        if isinstance(obj, dict):
            return {k: self._round_floats_in(v, ndigits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._round_floats_in(v, ndigits) for v in obj]
        if isinstance(obj, float):
            return round(obj, ndigits)
        return obj

    _FILTER_TOL = 1e-6  # 落盘前清理：权重/收益率在此范围内视为 0 或 1

    def _should_skip_perf_entry(self, data: dict) -> bool:
        """
        是否跳过该条不落盘：除最后一项（现金）外权重均为 0，且 performance[0]（收益率）为 0 的条目剔除。
        适用于任意 n，不阻塞主进程（仅主线程内轻量计算）。
        """
        try:
            weights = data.get("decision_weights")
            perf = data.get("performance")
            if not weights or len(weights) < 2 or not perf or len(perf) < 1:
                return False
            weights = list(weights) if isinstance(weights, tuple) else weights
            tol = self._FILTER_TOL
            non_cash_all_zero = all(abs(float(w)) <= tol for w in weights[:-1])
            cash_near_one = abs(float(weights[-1]) - 1.0) <= tol
            rtr_near_zero = abs(float(perf[0])) <= tol
            return bool(non_cash_all_zero and cash_near_one and rtr_near_zero)
        except (TypeError, ValueError, IndexError):
            return False

    def save_meta(self):
        """保存 meta 到本地（结构见 pool.py 顶部：顶层固定 + train_config 随机）"""
        meta_path = self._get_base_path() / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(DATA_CACHE_POOL.get_meta(), f, ensure_ascii=False)
        

    def _write_jsonl_worker(self, record_path: Path, lines: list):
        """后台线程：将已序列化的行追加写入；写完后异步触发发送 perf 与 record（发送成功则删本次 perf）。"""
        try:
            with open(record_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            logger.debug(f"追加记录快照: {record_path}, 条数={len(lines)}")
            threading.Thread(target=self.send_perf_and_record, daemon=True).start()
        except Exception as e:
            logger.error(f"异步写入 record.jsonl 失败: {e}")

    def append_performance_and_reward_snapshot(self, segment: bool = False, daemon: bool = True):
        """
        异步保存表现和奖励快照到本地，按 JSONL 追加，不阻塞主流程。
        segment: True 时 cursor+=1 后写入新段文件，否则追加到当前段文件。
        """
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.warning("当前窗口未设置，跳过保存快照")
            return
        year, month = current_ym
        incremental_result = REWARD_MANAGER.get_incremental_snapshot(year, month)
        if not incremental_result:
            return
        lines = []
        for key, data in incremental_result.items():
            if self._should_skip_perf_entry(data):
                continue
            y, m, portfolio = key
            obj = {"year": y, "month": m, "portfolio": list(portfolio), "data": self._serialize_perf(data)}
            obj = self._round_floats_in(obj, 4)
            lines.append(json.dumps(obj, ensure_ascii=False))
        with self._snapshot_lock:
            if segment:
                self._performance_and_reward_snapshot_cursor += 1
            file_name = f"performance_and_reward_{self._performance_and_reward_snapshot_cursor}.jsonl"
            record_path = self._get_base_path() / file_name
        threading.Thread(target=self._write_jsonl_worker, args=(record_path, lines), daemon=daemon).start()

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

    def send_perf_and_record(self):
        """
        只发送 record.json 与 performance_and_reward_*.jsonl 到主机；发送成功后删除本次发送的 perf 文件。
        加锁保证「列清单 + rsync + 删本次列表」原子，避免其他落盘线程刚写的文件被误删。
        task_id / node_id 缺失时不发送、不删文件。
        """
        task_id = DATA_CACHE_POOL.get_task_id()
        node_id = DATA_CACHE_POOL.get_node_id()
        if not task_id or not node_id:
            logger.warning("task_id 或 node_id 为空，跳过发送 perf 与 record")
            return
        base = self._get_base_path()
        if not base.exists():
            return
        try:
            with self._send_lock:
                perf_files = sorted(base.glob("performance_and_reward_*.jsonl"))
                to_send = list(perf_files)
                src = str(base) + "/"
                dest = f"nodeuser@43.139.192.176::node_result/{task_id}/{node_id}/"
                cmd = [
                    "rsync", "-avz",
                    "--include=record.json",
                    "--include=performance_and_reward_*.jsonl",
                    "--exclude=*",
                    "--password-file=/Node/rsync.passwd",
                    "--port=8730",
                    src,
                    dest,
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if ret.returncode == 0:
                    for p in to_send:
                        try:
                            p.unlink()
                        except OSError as e:
                            logger.warning(f"发送成功后删除 perf 文件失败: {p}, {e}")
                    logger.debug(f"发送 perf 与 record 成功，已删除 {len(to_send)} 个 perf 文件")
                else:
                    logger.warning(f"发送 perf 与 record 失败: returncode={ret.returncode}, stderr={ret.stderr!r}")
        except subprocess.TimeoutExpired:
            logger.warning("发送 perf 与 record 超时，未删除本地 perf 文件")
        except Exception as e:
            logger.exception(f"发送 perf 与 record 异常: {e}")

SAVER = Saver()

