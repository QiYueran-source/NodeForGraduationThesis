#%%
# 设置 Python 路径
import time
import src.utils.set.set_pypath

from src.worker.cache import DATA_CACHE_POOL


def setup_worker_pool():
    """初始化 Worker 缓存池：仅 meta，train 数据由数据线程从主节点加载"""

    DATA_CACHE_POOL.put_stock_list(['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010'])
    DATA_CACHE_POOL.put_N(10)
    n_factors = len(DATA_CACHE_POOL.get_factors_list())
    DATA_CACHE_POOL.put_train_config({
        'n': 2,
        'm': 2,
        'max_portfolios_num': 3,
        'mask': [1] * n_factors,
        'performance_config': {
            'risk_free_rate': 0.02,
        },
    })
    DATA_CACHE_POOL.put_start_year(1997)
    DATA_CACHE_POOL.put_earliest_year_month(1997, 1)
setup_worker_pool()

import src.worker.data.data_thread as data_thread
data_thread.data_thread_start()
time.sleep(30)  

# 测试ADAPTER 
from src.worker.agent.data.adapter import AGENT_DATA_ADAPTER  

x,y = AGENT_DATA_ADAPTER.get_train_data(1997, 1, '000001')
print(x,y)
#%%