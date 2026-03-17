"""
MLP 预测相关模块。

该包中的实现仅在 train_config.use_mlp_predict=True 时由 worker.train 调用，
用于替代 sb3 强化学习的决策生成逻辑。
"""

