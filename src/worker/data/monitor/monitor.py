"""
数据监控器   
1.定时监控pool中的数据  
2.如果年份不够，则调用loader加载  
"""  
# 库  
from logging import CRITICAL
from re import T
import time 
import threading  
import yaml  
from typing import Optional

# 组件 
from src.worker.data.loader import DATA_LOADER  
from src.worker.cache import DATA_CACHE_POOL  

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='DataMonitor')

class DataMonitor:
    def __init__(self):
        # 当前年份
        self.now_year = None

        # 启动标志和线程管理
        self.started = False
        self.monitor_thread = None

        # 配置
        self._config = {}

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/hyparam.yaml', 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f).get('data_thread', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise 
        
        self.now_year = DATA_CACHE_POOL.get_start_year() - 1 # 比如1997年开始，则设置当前为1996
        if not self.now_year:
            logger.error("未设置开始年份")
            raise Exception("未设置开始年份")
        # 当前窗口由 adapter 通过 record 维护，monitor 仅用 cache 进度（current_train_year_month）判断加载
        logger.debug(f"_load_config 完成: start_year={self.now_year}, check_interval={self._config.get('check_interval')}, min_year={self._config.get('min_year')}, max_year={self._config.get('max_year')}")

    def monitor_loop(self):
        """
        监控数据主循环
        定期检查数据状态，如果数据不足则自动加载
        """
        logger.info('数据监控循环启动')

        try:
            while self.started:
                try:
                    time.sleep(self._config.get('check_interval', 10))

                    # 检查是否仍在运行（防止虚假唤醒）
                    if not self.started:
                        break
                    
                    # 同步执行，避免异步执行导致数据不一致  
                    self._check_and_load()

                except Exception as e:
                    logger.error(f'数据监控循环中发生错误: {e}')
                    # 短暂等待后继续，避免错误循环
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info('收到中断信号，停止数据监控')

        logger.info('数据监控循环结束')

    def _load_year_with_retry(self, year: int, exception_retry_times: int = 3, retry_interval: int = 5, retry_timeout: int = 300) -> Optional[dict]:
        """
        按配置对单个年份的数据加载做重试控制：
        - data is None：认为是“异常”，使用 exception_retry_times 次数控制
        - data == {}：认为是“当前无数据”，使用 retry_timeout + retry_interval 的时间窗口控制
        - 有数据：立即返回
        """
        start_ts = time.time()
        attempt = 0               # 总尝试次数（统计用）
        exception_count = 0       # 仅统计 data is None 的次数

        while True:
            attempt += 1
            data = DATA_LOADER.fetch_data(year)

            # 有数据，直接成功
            if data:
                logger.info(
                    f"year={year} 加载成功，条数={len(data)}，尝试={attempt} 次，异常次数={exception_count}"
                )
                return data

            # data is None -> 异常路径，走“次数”控制
            if data is None:
                exception_count += 1
                if exception_count >= exception_retry_times:
                    logger.error(
                        f"year={year} 加载异常重试次数达到上限 "
                        f"{exception_retry_times} 次，最后一次在第 {attempt} 次尝试"
                    )
                    # 抛异常，让外层 monitor_loop 捕获并按已有逻辑 sleep(5)
                    raise Exception(
                        f"year={year} 数据加载异常重试次数超限 "
                        f"({exception_count}/{exception_retry_times})"
                    )

                logger.warning(
                    f"year={year} 加载异常(返回 None)，"
                    f"{retry_interval}s 后重试 "
                    f"(第 {attempt} 次，总异常次数 {exception_count}/{exception_retry_times})"
                )
                time.sleep(retry_interval)
                continue

            # 走到这里说明 data == {} -> 当前无数据，走“时间窗口”控制
            elapsed = time.time() - start_ts
            if elapsed >= retry_timeout:
                logger.error(
                    f"year={year} 当前无数据重试超时，已尝试 {attempt} 次，"
                    f"耗时 {elapsed:.1f}s (retry_timeout={retry_timeout})"
                )
                raise Exception(
                    f"year={year} 当前无数据重试超时 "
                    f"(elapsed={elapsed:.1f}s, retry_timeout={retry_timeout})"
                )

            logger.warning(
                f"year={year} 当前无数据(空 dict)，"
                f"{retry_interval}s 后重试 "
                f"(第 {attempt} 次，已耗时 {elapsed:.1f}s)"
            )
            time.sleep(retry_interval)

    def _check_and_load(self):
        """检查数据：以 cache 中最大 (year, month) 为进度，不足则加载"""
        current_ym = DATA_CACHE_POOL.current_train_year_month
        if not current_ym:
            logger.error("无法获取缓存加载进度 (year, month)")
            raise Exception("无法获取缓存加载进度 (year, month)")
        now_year = current_ym[0]

        years = DATA_CACHE_POOL.train_years_count
        min_year = self._config.get('min_year', 1)
        max_year = self._config.get('max_year', 2)

        if years <= min_year:
            need_load_years = max_year - years
            logger.info(f"年份数量不足，需要加载数据: years={years} <= min_year={min_year}, 将加载 {need_load_years} 年")

            for year in range(now_year + 1, now_year + need_load_years + 1):
                logger.debug(f"开始从 loader 拉取 year={year}")

                # 边界判断
                end_year = DATA_CACHE_POOL.get_end_year() or 2024 
                if year > end_year:
                    logger.info(f"已到达结束年份，停止加载: year={year} > end_year={end_year}")
                    break

                # 使用带重试机制的加载方法
                data = self._load_year_with_retry(
                    year,
                    exception_retry_times=self._config.get('exception_retry_times', 3),
                    retry_interval=self._config.get('retry_interval', 5),
                    retry_timeout=self._config.get('retry_timeout', 300),
                )

                logger.debug(f"loader 返回 year={year} 条数={len(data)}")
                items = [{'code': code, 'year': year, 'month': month, 'data': data} for (month, code), data in data.items()]
                DATA_CACHE_POOL.batch_put_train(items)
                DATA_CACHE_POOL.put_record('latest_data', {"year": year, "count": len(items)})
                self.now_year = year
                # DATA_CACHE_POOL.put_current_year_month(year, 12) # 废弃，进度由agent的rolling控制 
                logger.info(f"数据已写入缓存池: year={year}, 条数={len(items)}, 当前窗口=({year}, 12)")  

    def start(self):
        """
        启动数据监控器
        创建并启动监控线程，定期检查数据状态
        """
        if self.started:
            logger.info('数据监控器已经启动')
            return

        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning('监控线程仍在运行中')
            return

        try:
            # 创建并启动监控线程
            self.started = True
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop,
                name='DataMonitor',
                daemon=True
            )
            self.monitor_thread.start()

            logger.info('数据监控器启动成功')

        except Exception as e:
            self.started = False
            logger.error(f'启动数据监控器失败: {e}')
            raise

    def stop(self):
        """
        停止数据监控器
        优雅地停止监控线程并等待其结束
        """
        if not self.started:
            logger.info('数据监控器已关闭')
            return

        logger.info('正在停止数据监控器...')

        try:
            # 设置停止标志
            self.started = False

            # 等待线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.debug('等待监控线程结束...')
                self.monitor_thread.join(timeout=10)  # 最多等待10秒

                if self.monitor_thread.is_alive():
                    logger.warning('监控线程未在规定时间内结束')
                else:
                    logger.debug('监控线程已正常结束')

            # 清理线程引用
            self.monitor_thread = None

            logger.info('数据监控器已停止')

        except Exception as e:
            logger.error(f'停止数据监控器时发生错误: {e}')
            # 即使出错也要确保状态正确
            self.started = False
            self.monitor_thread = None

    def is_running(self) -> bool:
        """
        检查数据监控器是否正在运行
        :return: 是否正在运行
        """
        return (self.started and
                self.monitor_thread and
                self.monitor_thread.is_alive())
        # 剩余代码  TODO  


DATA_MONITOR = DataMonitor() 
    

    




    