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
        # 初始化当前窗口为 start_year 年 1 月
        DATA_CACHE_POOL.put_current_year_month(self.now_year, 1)
        logger.debug("_load_config 完成: start_year=%s, check_interval=%s, min_year=%s, max_year=%s",
                    self.now_year, self._config.get('check_interval'), self._config.get('min_year'), self._config.get('max_year'))

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

                    self._check_and_load()

                except Exception as e:
                    logger.error(f'数据监控循环中发生错误: {e}')
                    # 短暂等待后继续，避免错误循环
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info('收到中断信号，停止数据监控')

        logger.info('数据监控循环结束')

    def _check_and_load(self):
        """检查数据"""
        current_ym = DATA_CACHE_POOL.get_current_year_month()
        if not current_ym:
            logger.error("未设置当前窗口 (year, month)")
            raise Exception("未设置当前窗口 (year, month)")
        now_year = current_ym[0]

        years = DATA_CACHE_POOL.count_years_train()
        min_year = self._config.get('min_year', 1)
        max_year = self._config.get('max_year', 2)

        if years <= min_year:
            need_load_years = max_year - years
            logger.info("年份数量不足，需要加载数据: years=%d <= min_year=%d, 将加载 %d 年", years, min_year, need_load_years)
            for year in range(now_year + 1, now_year + need_load_years + 1):
                logger.debug("开始从 loader 拉取 year=%d", year)
                data = DATA_LOADER.fetch_data(year)
                logger.debug("loader 返回 year=%d 条数=%d", year, len(data))
                items = [{'code': code, 'year': year, 'month': month, 'data': data} for (month, code), data in data.items()]
                DATA_CACHE_POOL.batch_put_train(items)
                self.now_year = year
                DATA_CACHE_POOL.put_current_year_month(year, 12)
                logger.info("数据已写入缓存池: year=%d, 条数=%d, 当前窗口=(%d, 12)", year, len(items), year)  

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
    

    




    