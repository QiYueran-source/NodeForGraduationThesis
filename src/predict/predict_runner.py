from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.data import AGENT_DATA_ADAPTER
from src.worker.agent.env import REWARD_MANAGER
from src.worker.save import SAVER
from src.utils.logger import get_module_logger

from src.predict.mlp_model import MLPTrainer


logger = get_module_logger(__name__, prefix='[MLPPredict]')

CHECKPOINT_PATH = Path("/Node/checkpoint_mlp.safetensors")


def run_mlp_training_and_prediction() -> None:
    """
    使用 MLP 进行窗口级训练与预测，替代 sb3 的滚动预测阶段。
    - 仅在 train_config.use_mlp_predict=True 时由 worker.train 调用。
    - 停止条件：DATA_CACHE_POOL.get_running() 为 False 或 year > end_year。
    """
    tc = DATA_CACHE_POOL.get_train_config() or {}
    m = int(tc.get("m", 12))
    mask_len = int(tc.get("mask_len", 60))
    hidden_dim = int(tc.get("mlp_hidden_dim", 128))
    lr = float(tc.get("mlp_lr", 1e-3))
    seed = int(tc.get('seed', 42))
    epochs = int(tc.get("mlp_epochs", 20))
    batch_size = int(tc.get("mlp_batch_size", 32))

    input_dim = mask_len  # 当前实现假定 n=1，大多数场景成立（X 每行 = 1*mask_len 展平）
    trainer = MLPTrainer(input_dim=input_dim, hidden_dim=hidden_dim, lr=lr)

    # 获取因子掩码
    full_dim = 88
    rng = np.random.default_rng(seed)
    mask = np.zeros(full_dim, dtype=np.int32)
    selected_indices = rng.choice(full_dim, size=mask_len, replace=False)
    mask[selected_indices] = 1
    factor_mask = mask.tolist()  # factor_mask: List[int]，长度 88，含 mask_len 个 1
    factor_mask_bool = np.array(factor_mask, dtype=bool)

    use_checkpoint = bool(DATA_CACHE_POOL.get_checkpoint())
    if use_checkpoint and CHECKPOINT_PATH.exists():
        logger.info(f"加载 MLP checkpoint: {CHECKPOINT_PATH}")
        try:
            trainer.load_safetensor(str(CHECKPOINT_PATH))
        except Exception as e:
            logger.warning(f"加载 MLP checkpoint 失败，将从头训练: {e}")

    logger.info("========= 开始 MLP 窗口训练+预测 =========")
    _save_cursor = 0
    ec = DATA_CACHE_POOL.get_env_config() or {}
    save_record_every_n_steps = ec.get("save_record_every_n_steps", 50)

    while DATA_CACHE_POOL.get_running():
        ym = DATA_CACHE_POOL.get_current_year_month()
        if ym is None:
            logger.warning("当前窗口未设置，结束 MLP 预测")
            break
        year, month = ym
        if year > (DATA_CACHE_POOL.get_end_year() or year):
            logger.info(f"MLP 预测结束(已到 end_year), year={year}")
            SAVER.save_record()
            SAVER.append_performance_and_reward_snapshot(segment=True)
            break
        
        all_portfolios = AGENT_DATA_ADAPTER.get_portfolio_pool() 
        all_codes = [p[0] for p in all_portfolios] # 单个证券
        logger.debug(f"获取{ym}训练窗口的组合: {all_codes}")
        
        # 构造监督数据（一个窗口内使用 0..m-2 期的因子预测下一期组合收益）
        _ym = AGENT_DATA_ADAPTER._roll_year_month(ym, -m+1)
        end_ym = AGENT_DATA_ADAPTER._roll_year_month(ym, -1)
        while not AGENT_DATA_ADAPTER._year_month_greater(_ym, end_ym):
            y,m = _ym
            train_data = [AGENT_DATA_ADAPTER.get_train_data(y, m, code) for code in all_codes]
            if len(train_data) >= 3:
                # 过滤掉无效数据
                valid_train_data = [(factors, next_return) for factors, next_return in train_data
                                  if factors is not None and next_return is not None]
                if len(valid_train_data) >= 3:
                    logger.debug(f"获取{ym}训练窗口的因子数据, valid_train_data: 数量: {len(valid_train_data)}")
                    X_train = []
                    y_train = []
                    for factors, next_return in valid_train_data:
                        # 如果 factors 是 numpy 数组：
                        factors = np.asarray(factors)
                        factors_selected = factors[factor_mask_bool]  # shape = (mask_len,)
                        X_train.append(factors_selected)
                        y_train.append(next_return)
                    trainer.fit(X_train, y_train)
                else:
                    logger.warning(f"获取{ym}训练窗口的因子数据不足, valid_train_data: 数量: {len(valid_train_data)}")
                    _ym = AGENT_DATA_ADAPTER._roll_year_month(_ym, 1)
                    continue
            else:
                logger.warning(f"获取{ym}训练窗口的因子数据不足, train_data: 数量: {len(train_data)}")
                _ym = AGENT_DATA_ADAPTER._roll_year_month(_ym, 1)
                continue
            _ym = AGENT_DATA_ADAPTER._roll_year_month(_ym, 1)
        
        # 获取ym，all_codes的因子数据，用于预测 
        valid_codes = []
        factors_list = []
        real_return_list = []
        for code in all_codes:
            data = AGENT_DATA_ADAPTER.get_train_data(year, month, code)
            if data is None:
                logger.warning(f"获取{ym}训练窗口的因子数据为空, code: {code}, data: {data}")
                continue
            factors, real_return = data
            factors = np.asarray(factors)
            factors_selected = factors[factor_mask_bool]  # shape = (mask_len,)
            factors_list.append(factors_selected)
            valid_codes.append(code)
            real_return_list.append(real_return)
        if len(factors_list) == 0:
            logger.warning(f"获取{ym}训练窗口的因子数据为空, factors_list: 数量: {len(factors_list)}")
            continue
        logger.debug(f"开始预测{ym}训练窗口的因子数据")
        predictions = trainer.predict(factors_list)
        for code, prediction, real_return in zip(valid_codes, predictions, real_return_list):
            REWARD_MANAGER.record_predict_and_real_return(
                year, month, code, prediction, real_return
            )

        # 滚动窗口 + 定期保存
        logger.debug(f"滚动窗口 + 定期保存, _save_cursor: {_save_cursor}")
        AGENT_DATA_ADAPTER.win_roll()
        _save_cursor += 1
        if (
            _save_cursor % save_record_every_n_steps == 0
            and _save_cursor >= save_record_every_n_steps
        ):
            SAVER.save_record()

        try:
            logger.info(f"保存 MLP checkpoint 到 {CHECKPOINT_PATH}")
            trainer.save_safetensor(str(CHECKPOINT_PATH))
        except Exception as e:
            logger.warning(f"保存 MLP checkpoint 失败: {e}")

    logger.info("========= MLP 窗口训练+预测结束 =========")


