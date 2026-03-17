from pathlib import Path
from typing import Tuple

import numpy as np

from src.worker.cache.pool import DATA_CACHE_POOL
from src.worker.agent.data import AGENT_DATA_ADAPTER
from src.worker.agent.env import REWARD_MANAGER
from src.worker.save import SAVER
from src.utils.logger import get_module_logger

from src.predict.mlp_model import MLPTrainer
from src.predict.dataset_builder import build_supervised_dataset_for_window


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
    epochs = int(tc.get("mlp_epochs", 20))
    batch_size = int(tc.get("mlp_batch_size", 32))

    input_dim = mask_len  # 当前实现假定 n=1，大多数场景成立
    trainer = MLPTrainer(input_dim=input_dim, hidden_dim=hidden_dim, lr=lr)

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

        portfolio = AGENT_DATA_ADAPTER.win_get_a_portfolio()
        if not portfolio:
            logger.info("无可用 portfolio，结束 MLP 预测")
            SAVER.save_record()
            SAVER.append_performance_and_reward_snapshot(segment=True)
            break

        portfolio_tuple = tuple(portfolio)

        # 构造监督数据
        X_train, y_train = build_supervised_dataset_for_window(
            ym, m, portfolio_tuple
        )
        if X_train is not None and len(X_train) >= 3:
            logger.debug(
                f"MLP 训练样本: year={year}, month={month}, "
                f"samples={len(X_train)}, input_dim={input_dim}, "
                f"y_mean={float(y_train.mean()):.6f}, y_std={float(y_train.std()):.6f}"
            )
            trainer.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        else:
            logger.warning(
                f"样本不足或不可用，跳过本窗口 MLP 训练: ym={ym}, "
                f"samples={0 if X_train is None else len(X_train)}"
            )

        # 当前期输入 X_t: 使用当前窗口因子张量的最后一列 (最新一期)
        t = AGENT_DATA_ADAPTER.win_get_factors_tensor(portfolio_tuple)
        if t is None or t.numel() == 0:
            logger.warning(f"当前窗口因子为空，跳过预测: ym={ym}, portfolio={portfolio_tuple}")
            r_pred = 0.0
        else:
            factors = t.numpy().astype(np.float32)  # (n, m, mask_len)
            if factors.ndim != 3:
                logger.warning(
                    f"因子张量维度异常，期望 3 维，实际 {factors.ndim}，跳过预测: ym={ym}"
                )
                r_pred = 0.0
            else:
                n, m_obs, mask_len_obs = factors.shape
                if mask_len_obs != mask_len:
                    logger.warning(
                        f"因子维度与 mask_len 不一致: {mask_len_obs}!={mask_len}，跳过预测: ym={ym}"
                    )
                    r_pred = 0.0
                else:
                    # 取时间维最后一期的因子 (n, mask_len)，展平
                    x_t = factors[:, m_obs - 1, :].reshape(-1)
                    r_pred_raw = float(trainer.predict(x_t)[0])
                    # 百分化，例如 0.012 -> 1.2
                    r_pred = r_pred_raw * 100.0
                    logger.debug(
                        f"MLP 预测: ym={ym}, portfolio={portfolio_tuple}, "
                        f"r_pred_raw={r_pred_raw:.6f}, r_pred_percent={r_pred:.4f}"
                    )

        _apply_decision_with_pred(portfolio_tuple, ym, r_pred)

        # 滚动窗口 + 定期保存
        AGENT_DATA_ADAPTER.win_roll()
        _save_cursor += 1
        if (
            _save_cursor % save_record_every_n_steps == 0
            and _save_cursor >= save_record_every_n_steps
        ):
            SAVER.save_record()

    if use_checkpoint:
        try:
            logger.info(f"保存 MLP checkpoint 到 {CHECKPOINT_PATH}")
            trainer.save_safetensor(str(CHECKPOINT_PATH))
        except Exception as e:
            logger.warning(f"保存 MLP checkpoint 失败: {e}")

    logger.info("========= MLP 窗口训练+预测结束 =========")


def _apply_decision_with_pred(
    portfolio: Tuple[str, ...], ym: Tuple[int, int], r_pred: float
) -> None:
    """
    将预测收益 r_pred 映射到 decision_weights 与 reward。
    当前实现假设 n=1：全仓持有该组合（不持有现金），reward 保存为预测收益百分值。
    """
    year, month = ym
    # 简单方案：n=1，全仓该证券，现金 0
    decision_weights = [1.0, 0.0]
    try:
        REWARD_MANAGER.record_decision_and_predicted_return(
            year, month, portfolio, decision_weights, r_pred
        )
    except Exception as e:
        logger.warning(
            f"记录 MLP 决策与预测收益失败: ym={ym}, portfolio={portfolio}, err={e}"
        )

