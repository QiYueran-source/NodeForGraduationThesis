# 设置 Python 路径
import time
import src.utils.set.set_pypath

from src.worker.cache import DATA_CACHE_POOL


def setup_worker_pool():
    """初始化 Worker 缓存池：仅 meta，train 数据由数据线程从主节点加载"""
    DATA_CACHE_POOL.put_stock_list(['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010'])
    DATA_CACHE_POOL.put_N(5)
    n_factors = len(DATA_CACHE_POOL.get_factors_list())
    DATA_CACHE_POOL.put_train_config({
        'n': 2,
        'm': 2,
        'max_portfolios_num': 3,
        'mask': [1] * n_factors,
    })
    DATA_CACHE_POOL.put_start_year(1997)
    DATA_CACHE_POOL.put_earliest_year_month(1997, 1)


def _roll_year_month(ym, months):
    """年月滚动，不依赖适配器。ym=(year, month), months 为滚动月数。"""
    y, m = ym
    total = y * 12 + (m - 1) + months
    return (total // 12, total % 12 + 1)


def test_rolling_half_year():
    """
    测试数据线程：一次加载 1 年数据，每次加载后 sleep 30s，再获取下一年。
    无数据时也等待 30s 后重试。仅使用 DATA_CACHE_POOL，不依赖适配器。
    """
    print('--- 数据线程轮转测试（每年一批） ---')
    stock_list = DATA_CACHE_POOL.get_stock_list()
    start_year = DATA_CACHE_POOL.get_start_year() or 1997
    start_ym = (start_year, 1)
    year_months = 12
    sample_size = 3
    wait_seconds = 30

    while True:
        collected = []
        for i in range(year_months):
            y, m = _roll_year_month(start_ym, i)
            for code in stock_list:
                data = DATA_CACHE_POOL.get_train(code, y, m)
                if data is not None:
                    collected.append((y, m, code, data))

        count = len(collected)
        current_year = start_ym[0]
        print(f'\n一年 {current_year} 共获取 {count} 条')
        if current_year > start_year:
            prev_year = current_year - 1
            prev_in_cache = DATA_CACHE_POOL.contains_train(prev_year)
            print(f'  上一年 {prev_year} 在缓存: {"是" if prev_in_cache else "否"}')

        if count == 0:
            print(f'本年度无数据，{wait_seconds}s 后重试')
            time.sleep(wait_seconds)
            continue

        # 打印样本
        for j, (y, m, code, data) in enumerate(collected[:sample_size]):
            factors, rtr = data[0], data[1]
            n_f = len(factors) if factors else 0
            head = factors[:3] if factors and len(factors) >= 3 else (factors or [])
            print(f'  样本{j+1}: ({y}, {m}, {code}) 因子数={n_f} 收益={rtr} 因子前3={head}')
        if count > sample_size:
            print(f'  ... 其余 {count - sample_size} 条')

        time.sleep(wait_seconds)
        start_ym = _roll_year_month(start_ym, year_months)


if __name__ == '__main__':
    setup_worker_pool()
    from src.worker.data.data_thread import data_thread_start, data_thread_stop

    data_thread_start()
    time.sleep(5)
    test_rolling_half_year()
    data_thread_stop()
