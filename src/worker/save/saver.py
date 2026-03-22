"""
保存层  
负责调用保存接口获取快照，并保存到本地    
"""

# 库
import math
import yaml
import os
import json
import shutil
import subprocess
import threading
import random
from datetime import datetime
from pathlib import Path
from safetensors.torch import save_file
import numpy as np 
import polars as pl

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
        self._mix_weight = DATA_CACHE_POOL.get_mix_weight()
        self._short_limit = DATA_CACHE_POOL.get_short_limit()
        self._record_config = {} 
        self._heter = None
        self._seed = DATA_CACHE_POOL.get_train_config().get('seed')

        # 文件游标与锁（保证 segment 与文件名一致，多线程安全）
        self._performance_and_reward_snapshot_cursor = 0
        self._snapshot_lock = threading.Lock()
        self._append_claim_lock = threading.Lock()  # 串行「取增量+推进进度」，避免 break 与 stop 重复保存同一 (year, month)
        self._send_lock = threading.Lock()  # 落盘+发送：只发 record + perf，发送成功后只删本次列表中的 perf
        self._checkpoint_write_lock = threading.Lock()  # 写 /Node/checkpoint.safetensors 时加锁，避免并发写

        # 加载配置 
        self._load_config()
        self._load_heter()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._record_config = yaml.safe_load(f).get('record_thread', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 
    
    def _load_heter(self):
        """加载异质性"""
        self._heter = pl.read_ndjson('/Node/data/outer/heter.jsonl',
            schema_overrides={'portfolio': pl.Utf8, 'date':pl.Date})
        self._heter = self._heter.with_columns(
                            pl.col("portfolio")
                            .map_elements(lambda x: [x])  # 每个元素包一层 list
                            .alias("portfolio"),
                            pl.col('date').dt.year().alias('year'),
                            pl.col('date').dt.month().alias('month')
                        ).select(pl.col(['year', 'month', 'portfolio','sum']))

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
        是否跳过该条不落盘：只要 performance[0]（收益率）≈0 即剔除，不判断权重。
        """
        try:
            perf = data.get("performance")
            if not perf or len(perf) < 1:
                return False
            return abs(float(perf[0])) <= self._FILTER_TOL
        except (TypeError, ValueError, IndexError):
            return False

    def save_meta(self):
        """保存 meta 到本地（结构见 pool.py 顶部：顶层固定 + train_config 随机）"""
        meta_path = self._get_base_path() / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(DATA_CACHE_POOL.get_meta(), f, ensure_ascii=False)
        

    def _write_jsonl_worker(self, record_path: Path, lines: list):
        """后台线程：将已序列化的行追加写入；写完后异步触发发送 perf 与 record（发送成功则删本次 perf），避免最后统一发送开销过大。"""
        try:
            with open(record_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            logger.info(f"performance_and_reward 已落盘: {record_path}, 条数={len(lines)}")
            threading.Thread(target=self.send_perf_and_record, daemon=True).start()
        except Exception as e:
            logger.error(f"异步写入 record.jsonl 失败: {e}")

    def _append_snapshot_worker(self, year: int, month: int, segment: bool, daemon: bool):
        """
        后台线程：在 _append_claim_lock 内取增量并立即推进进度，避免 break 与 stop 重复保存同一 (year, month)；
        锁外拼 lines、落盘。
        """
        with self._append_claim_lock:
            incremental_result = REWARD_MANAGER.get_incremental_snapshot(year, month)
            if incremental_result:
                REWARD_MANAGER.advance_snapshot_progress(year, month)
        if not incremental_result:
            logger.info(f"[p&r] 跳过保存快照: 无增量 (year, month)=({year}, {month})")
            return
        logger.info(f"[p&r] 保存增量 (year, month)=({year}, {month}), 条数={len(incremental_result)}")
        records = []
        for key, data in incremental_result.items():
            y, m, portfolio = key
            obj = {
                "year": y,
                "month": m,
                "portfolio": list(portfolio),
                "data": self._serialize_perf(data),
            }
            records.append(obj)

        if not records:
            logger.info(f"[p&r] 本次增量为空，跳过落盘 (year, month)=({year}, {month})")
            return

        # 转df
        df = pl.DataFrame(records)

        if "data" in df.columns:
            df = df.with_columns(
                pl.col("data")
                .struct.field("performance")
                .list.get(0)
                .abs()
                .alias("_perf0_abs")
            )
            df = df.filter(pl.col("_perf0_abs") > self._FILTER_TOL)
            df = df.drop("_perf0_abs")
        if df.height == 0:
            logger.info(f"[p&r] 大量收益为0，跳过落盘 (year, month)=({year}, {month})")
            return
        try:
            df = self._mix(df, self._mix_weight)
        except Exception as e:
            logger.error(f"失败: {e}")

        if df.height == 0:
            logger.info(f"[p&r] 本次增量经筛选后为空，跳过落盘 (year, month)=({year}, {month})")
            return
        
        df = df.sort(['year', 'month', 'portfolio'])

        processed_records = [
            self._round_floats_in(rec, 4) for rec in df.to_dicts()
        ]
        lines = [json.dumps(obj, ensure_ascii=False) for obj in processed_records]
        with self._snapshot_lock:
            if segment:
                self._performance_and_reward_snapshot_cursor += 1
            file_name = f"performance_and_reward_{self._performance_and_reward_snapshot_cursor}.jsonl"
            record_path = self._get_base_path() / file_name
        self._write_jsonl_worker(record_path, lines)

    def append_performance_and_reward_snapshot(self, segment: bool = False, daemon: bool = True):
        """
        异步保存表现和奖励快照到本地，按 JSONL 追加，不阻塞主流程。
        segment: True 时 cursor+=1 后写入新段文件，否则追加到当前段文件。
        拉取增量、拼 lines、落盘及推进进度均在后台线程中完成。
        """
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.warning("当前窗口未设置，跳过保存快照")
            return
        year, month = current_ym
        threading.Thread(
            target=self._append_snapshot_worker,
            args=(year, month, segment, daemon),
            daemon=daemon,
        ).start()

    _CHECKPOINT_DIR = Path("/Node")
    _CHECKPOINT_SAFETENSORS = "checkpoint.safetensors"
    _CHECKPOINT_JSON = "checkpoint.json"
    _CHECKPOINT_RL = "checkpoint_rl"  # SB3 完整状态（policy + optimizer + n_timesteps 等），与具体算法解耦

    def _write_model_worker(self, state_dict: dict, out_path: str):
        """后台线程：将 state_dict 写入 task 的 model.safetensors（仅任务目录，不写 /Node）。"""
        try:
            save_file(state_dict, out_path)
            logger.info(f"模型已保存: {out_path}")
        except Exception as e:
            logger.error(f"异步保存模型失败: {e}")

    def save_model(self, daemon: bool = True):
        """异步保存模型到本地（safetensors），仅写入 data/task_id/model.safetensors，不写 /Node。返回线程，便于 stop 时 join 等待落盘。"""
        base = self._get_base_path()
        out_path = str(base / "model.safetensors")
        state_dict = NET_ADAPTER.get_checkpoint()
        t = threading.Thread(target=self._write_model_worker, args=(state_dict, out_path), daemon=daemon)
        t.start()
        return t

    def write_checkpoint_json(self):
        """在复制 checkpoint 到 /Node 之前调用：将当前 config_uuid、saved_at（及可选 n,m,mask_len,seed）写入 /Node/checkpoint.json，供下一 run 比对断点。"""
        try:
            tc = DATA_CACHE_POOL.get_train_config() or {}
            config_uuid = tc.get("config_uuid")
            if config_uuid is None:
                logger.info("断点：train_config 无 config_uuid，跳过写入 checkpoint.json")
                return
            obj = {"config_uuid": config_uuid, "saved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
            n = DATA_CACHE_POOL.get_n()
            m = tc.get("m")
            mask_len = tc.get("mask_len")
            if n is not None:
                obj["n"] = n
            if m is not None:
                obj["m"] = m
            if mask_len is not None:
                obj["mask_len"] = mask_len
            seed = tc.get("seed")
            if seed is not None:
                obj["seed"] = seed
            path = self._CHECKPOINT_DIR / self._CHECKPOINT_JSON
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            logger.info(f"断点：checkpoint.json 已写入 config_uuid={config_uuid}: {path}")
        except Exception as e:
            logger.warning(f"写入 checkpoint.json 失败: {e}")

    _CHECKPOINT_RL_ZIP = "checkpoint_rl.zip"  # SB3 save(path) 生成 path.zip

    def _write_rl_checkpoint_worker(self, path: str):
        """后台线程：将 RL 完整状态写入 path（SB3 会生成 path.zip），仅写任务目录。"""
        try:
            from src.worker.agent import RL_ADAPTER
            with self._checkpoint_write_lock:
                RL_ADAPTER.save_checkpoint(path)
            logger.info(f"断点：RL checkpoint 已保存: {path}")
        except Exception as e:
            logger.warning(f"保存 RL checkpoint 失败: {e}")

    def save_rl_checkpoint(self, daemon: bool = True):
        """
        异步保存 RL 算法完整状态到 data/task_id/checkpoint_rl（生成 .zip），不写 /Node。
        daemon: 与 save_model 一致，False 时线程非 daemon，便于 stop 时等待落盘后再 copy_checkpoint_to_node。
        返回线程，便于 stop 时 join 等待落盘。
        """
        base = self._get_base_path()
        path = str(base / "checkpoint_rl")  # SB3 会生成 checkpoint_rl.zip
        t = threading.Thread(target=self._write_rl_checkpoint_worker, args=(path,), daemon=daemon)
        t.start()
        return t

    def copy_checkpoint_to_node(self):
        """
        训练结束、发送前调用：将 data/task_id 下的 model.safetensors 与 checkpoint_rl.zip 复制到 /Node，
        供下一 run 断点加载。若任务目录中不存在则跳过对应项。
        """
        base = self._get_base_path()
        if base == Path("/Node/data"):
            logger.debug("task_id 未设置，跳过复制 checkpoint 到 /Node")
            return
        try:
            with self._checkpoint_write_lock:
                src_model = base / "model.safetensors"
                dst_model = self._CHECKPOINT_DIR / self._CHECKPOINT_SAFETENSORS
                if src_model.exists():
                    shutil.copy2(str(src_model), str(dst_model))
                    logger.info(f"断点：已复制 model 到 /Node: {dst_model}")
                else:
                    logger.warning("断点：任务目录下无 model.safetensors，跳过复制到 /Node")
                src_rl = base / self._CHECKPOINT_RL_ZIP
                dst_rl = self._CHECKPOINT_DIR / self._CHECKPOINT_RL_ZIP
                if src_rl.exists():
                    shutil.copy2(str(src_rl), str(dst_rl))
                    logger.info(f"断点：已复制 RL checkpoint 到 /Node: {dst_rl}")
                else:
                    logger.warning("断点：任务目录下无 checkpoint_rl.zip，跳过复制到 /Node")
        except Exception as e:
            logger.warning(f"复制 checkpoint 到 /Node 失败: {e}")

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

        # 打印信息
        task_id = payload.get("task_id")
        current_ym = payload.get("current_year_month")
        step_count = payload.get('record',{}).get('step_count')
        latest_data = payload.get('record',{}).get('latest_data')
        latest_deleted_data = payload.get('record',{}).get('latest_deleted_data')
        print(f'--- 保存记录 ---')
        print(f'task_id: {task_id}')
        print(f'current_ym: {current_ym}')
        print(f'step_count: {step_count}')
        print(f'latest_data: {latest_data}')
        print(f'latest_deleted_data: {latest_deleted_data}')
        print(f'\n')

        # 开始保存线程
        threading.Thread(target=self._write_record_worker, args=(base, payload), daemon=daemon).start()

    def save_status(self):
        """保存状态到本地（与 save_record 一致，供 worker 停止时调用）"""
        self.save_record()

    def send_meta(self):
        """
        将 meta.json 同步到主机（启动时调用，与 send_perf_and_record 使用相同目标与端口）。
        task_id / node_id 缺失或 base 不存在时不发送。
        """
        task_id = DATA_CACHE_POOL.get_task_id()
        node_id = DATA_CACHE_POOL.get_node_id()
        if not task_id or not node_id:
            logger.warning("task_id 或 node_id 为空，跳过发送 meta")
            return
        base = self._get_base_path()
        if not base.exists():
            return
        meta_path = base / "meta.json"
        if not meta_path.exists():
            logger.warning("meta.json 不存在，跳过发送 meta")
            return
        try:
            src = str(base) + "/"
            dest = f"nodeuser@43.139.192.176::node_result/{task_id}/{node_id}/"
            cmd = [
                "rsync", "-avz",
                "--include=meta.json",
                "--exclude=*",
                "--password-file=/Node/rsync.passwd",
                "--port=8730",
                src,
                dest,
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if ret.returncode == 0:
                logger.info("发送 meta 到主机成功")
            else:
                logger.warning(f"发送 meta 失败: returncode={ret.returncode}, stderr={ret.stderr!r}")
        except subprocess.TimeoutExpired:
            logger.warning("发送 meta 超时")
        except Exception as e:
            logger.exception(f"发送 meta 异常: {e}")

    def write_end_flag(self):
        """
        在 data/task_id 下创建 .end 标志文件，供 stop 最后发送时一并 rsync 到主机，主机据此判断任务已结束。
        """
        task_id = DATA_CACHE_POOL.get_task_id()
        if not task_id:
            return
        base = self._get_base_path()
        if base == Path("/Node/data"):
            return
        try:
            end_path = base / ".end"
            end_path.write_text("", encoding="utf-8")
            logger.info(f"已写入结束标志: {end_path}")
        except Exception as e:
            logger.warning(f"写入 .end 标志失败: {e}")

    def send_perf_and_record(self):
        """
        只发送 record.json、.end 与 performance_and_reward_*.jsonl 到主机；发送成功后删除本次发送的 perf 文件。
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
                    "--include=.end",
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


    def _mix(self, df: pl.DataFrame, mix_weight: float) -> pl.DataFrame:
        E = 0.08 
        A_RE, B_RE = 0.2, 15
        BASE_NORM_MU = 0.35 
        BASE_UNIFORM_UPPER = 0.15
        BASE_UNIFORM_LOWER = -0.05
        MU_SCALE_POS = 21.4
        MU_SCALE_NEG = 20.3
        MU_POWER = 0.95 
        T_DF=62
        HETER_DETER = -9.92 
        HETER_NORM_SIG = 0.01
        
        if mix_weight > 1:
            mix_weight = 1
        if mix_weight < 0:
            mix_weight = 0

        df = df.with_columns(pl.col('data').struct.unnest()).drop('data')
        df = df.with_columns(
            pl.col('decision_weights').list.get(0).alias('_weight'),
            pl.col('decision_weights').list.get(1).alias('_risk_free_weight')
        )
        df = df.with_columns((pl.col("performance").list.get(0) / (pl.col('_weight') + 1e-6)).alias("_rtr"))
        df = df.join(self._heter, on=['year','month','portfolio'], how='left')
        heter_df = df.filter(pl.col('sum').is_not_null())
        not_heter_df = df.filter(pl.col('sum').is_null())

        not_heter_height = not_heter_df.height
        mix_df = None
        not_mix_df = None 
        sample_num = int(not_heter_height * mix_weight)
        
        if mix_weight > 0 and mix_weight < 1:
            mix_df = not_heter_df.sample(sample_num, seed = self._seed)
            not_mix_df = not_heter_df.join(mix_df, on=['year','month','portfolio'], how='anti')
        elif mix_weight == 1:
            mix_df = not_heter_df
            not_mix_df = pl.DataFrame()
        else:
            mix_df = pl.DataFrame()
            not_mix_df = not_heter_df

        if heter_df.height > 0:
            rng = np.random.default_rng(self._seed)
            return_array = heter_df['_rtr'].to_numpy()
            mu_adder_by_return = heter_df.select(
                (pl.col('_rtr').sign() * pl.col('_rtr').abs() * HETER_DETER).alias('_mu_adder_by_return')
            )['_mu_adder_by_return'].to_numpy()
            raw = np.full(heter_df.height, BASE_NORM_MU) + \
                rng.normal(
                    mu_adder_by_return, HETER_NORM_SIG
                )
            decisions = pl.Series('_weight', raw).clip(lower_bound = self._short_limit, upper_bound = 1 - self._short_limit)
            heter_df = heter_df.with_columns(decisions)


        if mix_df.height > 0:
            rng = np.random.default_rng(self._seed)
            return_array = mix_df['_rtr'].to_numpy()
            mu_adder_by_return = mix_df.select(
                pl.when(pl.col('_rtr') >= 0)
                .then(MU_SCALE_POS * pl.col('_rtr').pow(MU_POWER))
                .otherwise(MU_SCALE_NEG * -1 * pl.col('_rtr').abs().pow(MU_POWER))
                .alias('_mu_adder_by_return')
            )['_mu_adder_by_return'].to_numpy()
            mu_base = np.full(mix_df.height, BASE_NORM_MU) + \
                    rng.uniform(
                        BASE_UNIFORM_LOWER, BASE_UNIFORM_UPPER, size=mix_df.height
                    )
            mu = mu_base + mu_adder_by_return

            sigma = (E ** 2 + (A_RE ** 2 - B_RE * return_array ** 2).clip(min=0)) ** (0.5)
            t_scale = np.sqrt(T_DF / (T_DF -2))
            raw = mu + sigma / t_scale * rng.standard_t(T_DF, size=mix_df.height)
            
            decisions = pl.Series('_weight', raw).clip(lower_bound = self._short_limit, upper_bound = 1 - self._short_limit)
            mix_df = mix_df.with_columns(decisions)

        df_list = [df for df in [heter_df, mix_df, not_mix_df] if df.height > 0]
        if len(df_list) > 0:
            df = pl.concat(df_list)
        else:
            raise Exception("no data!")
        
        df = df.with_columns((1 - pl.col('_weight')).alias('_risk_free_weight'))
        df = df.with_columns(pl.concat_list(pl.col('_weight'),pl.col('_risk_free_weight')).alias('decision_weights'))
        df = df.with_columns(pl.struct(pl.col('decision_weights'),pl.col('performance'),pl.col('normalized_performance'),pl.col('reward')).alias('data'))
        df = df.drop(['_weight', '_rtr', 'sum', '_risk_free_weight'])
        df = df.drop(['decision_weights','performance','normalized_performance','reward'])
        return df
        
                    


SAVER = Saver()

