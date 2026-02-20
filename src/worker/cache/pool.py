"""
数据缓存池
线程安全的数据缓存，用于Worker进程内线程间通信  
包括训练数据、元数据和记录数据  
train: 由主机提供  
    - (code, year, month): [[因子],收益]   

meta：由主机提供，结构见下。约定：顶层 = 固定（环境统一），train_config = 随机（agent 异质性）。
【顶层 = 固定】
- task_id: 任务id
- start_year: 开始年份
- end_year: 结束年份（end_month 固定为 12，不单独提供接口）
- N: 总股票数量
- stock_list: 股票列表
- factors_list: 因子列表（避免麻烦，直接保存本地）
- earliest_year_month: 最早的年份和月份,(year, month)
- n: 一个组合中的证券数量（算上现金，共n+1个证券）
- max_portfolios_num: 对于总共n个证券，最多可构建组合数上限
- env_config: 环境配置
    - rf_end_year: 强化学习结束年份(后续年份不再学习但继续计算)，月份默认12
    - save_every_n_steps: 每多少步保存一次模型
    - sample_and_shuffle_seed: 采样与滚窗打乱种子；设后所有容器组合采样顺序、每窗口 shuffle 顺序一致，可复现
    - retrain_times: 同一窗口重复训练轮数，默认 1；>1 时本窗口组合用尽后重置游标并打乱再扫一轮，满轮后再滚窗
- performance_config: 表现计算配置
    - risk_free_rate: 无风险利率
    - （vol/sharpe/max_drawdown 的滚动窗口已统一为 train_config.m，不再使用 rolling_window / max_drawdown_window）
    - std_window: 标准化窗口期数
【train_config = 随机】
- seed: 随机种子
- m: 回看的期数
- mask_len: 因子掩码长度，默认60
- model_config: 模型配置（cate/dropout/config；设备有 GPU 则用 cuda，否则 cpu）
- reinforcement_config: 强化学习配置（rl_config/cate/opt: lr/clip_grad_norm/weight_decay）
- reward_config: 奖励配置（reward_weights: rtr/vol/sharpe/max_drawdown）

record: 由节点维护  
- running: 是否正在运行    
- current_year_month: 当前窗口(year,month)   
- pid: 进程号  
- node_id: 节点id (从 frp 状态文件中获取)  
"""

# 库
import threading
from typing import Dict, Optional, Tuple, Any, List
import datetime as dt 
import dateutil.relativedelta as dr 

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[DataCachePool]')

class DataCachePool:
    """线程安全的数据缓存池"""
    
    def __init__(self):
        """
        初始化缓存池
        """
        self._cache: Dict[Tuple[str, int, int], Any] = {} # 缓存数据：键为 (code, year, month)，值为数据
        
        # 结构化 meta（由主机提供）。约定：顶层键为固定，train_config 仅存随机部分，见本文件顶部注释。
        self._meta: Dict[str, Any] = {
            'task_id': None,
            'start_year': None,
            'N': None,
            'stock_list': [],
            'n': None,  # 固定，顶层。一个组合中的证券数量（算上现金共 n+1 个）
            'max_portfolios_num': None,  # 固定，顶层。可构建组合数上限
            'env_config': None,  # 固定，顶层。含 rf_end_year
            'performance_config': None,  # 固定，顶层。含 risk_free_rate；vol/sharpe/mdd 窗口用 train_config.m
            'factors_list': [
                                'absacc', 'acc', 'accp', 'ag', 'am', 'ato',
                                'beta', 'betad', 'betasq', 'bm', 'bm_ia', 'bveg',
                                'cash', 'cashpr', 'cfdebt', 'cfp', 'cfp_ia', 
                                'chfeps', 'chnanalyst', 'cinvest', 'coskew', 
                                'cr', 'crg', 'ct', 'depr', 'dp', 'ep', 'fgr5yr', 
                                'grcapx', 'grltnoa', 'hire', 'idskew', 'idvol', 
                                'illiq', 'imom', 'invchg', 'invest', 'invg', 'lagretn', 
                                'lev', 'lg', 'lm', 'mom12', 'mom36', 'mom6', 'momchg', 
                                'nanalyst', 'nincr', 'noa', 'ocfp', 'operprof', 
                                'pa', 'pchcapx_ia', 'pchdepr', 'pchgm_pchsale', 
                                'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 
                                'pchsaleinv', 'pmg', 'qr', 'qrg', 'rd_mve', 'rdsale', 
                                'realestate', 'roa', 'roe', 'roic', 'rsup', 'salecash', 
                                'saleinv', 'salerec', 'sfe', 'sg', 'sglnvg', 'sgr', 'size', 
                                'size_ia', 'skew', 'sp', 'std_dvol', 'std_turn', 'stdacc', 
                                'tang', 'taxchg', 'turn', 'vol', 'volumed'
                            ],
            'train_config': {  # 仅随机部分，结构见本文件顶部【train_config = 随机】
                'seed': None,
                'm': None,
                'mask_len': None,
                'model_config': None,
            },
        }

        # 记录数据（由节点维护），单独字典
        self._record: Dict[str, Any] = {
            'running': False,
            'current_year_month': None,
        }
        
        # 线程锁，保护缓存操作
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._record_lock = threading.Lock()
        
        logger.info("数据缓存池已创建")
    
    # ========== 训练数据接口 ==========
    def put_train(self, code: str, year: int, month: int, data: Any):
        """
        将数据放入缓存池
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :param data: 数据
        """
        key = (code, year, month)
        
        with self._lock:
            if key in self._cache:
                logger.warning(f"数据已存在，将被覆盖: {code}, {year}, {month}")
            self._cache[key] = data
            logger.debug(f"数据已放入缓存: {code}, {year}, {month}")
    
    def batch_put_train(self, items: List[Dict[str, Any]]):
        """
        批量将数据放入缓存池
        :param items: 数据列表，每个元素包含 code, year, month, data 字段
        """
        with self._lock:
            count = 0
            for item in items:
                code = item.get('code')
                year = item.get('year')
                month = item.get('month')
                data = item.get('data')
                
                if not all([code, year, month, data is not None]):
                    logger.warning(f"批量插入项格式错误，跳过: {item}")
                    continue
                
                key = (code, year, month)
                if key in self._cache:
                    logger.debug(f"批量插入：数据已存在，将被覆盖: {code}, {year}, {month}")
                
                self._cache[key] = data
                count += 1
            
            logger.info(f"批量插入完成，共插入 {count} 条数据")
    
    def get_train(self, code: str, year: int, month: int) -> Optional[Any]:
        """
        从缓存池获取数据并删除（剔除）
        :param code: 股票代码
        :param year: 年份
        :param month: 月份
        :return: 数据，[[因子],收益]
        """
        key = (code, year, month)
        
        with self._lock:
            if key in self._cache:
                data = self._cache.pop(key)  # 获取并删除
                logger.debug(f"数据已从缓存取出并删除: {code}, {year}, {month}")
                return data
            else:
                logger.debug(f"缓存中不存在: {code}, {year}, {month}")
                return None
    
    def contains_train(self, year: int, month: Optional[int] = None, code: Optional[str] = None) -> bool:
        """
        检查数据是否存在（支持部分匹配）
        :param year: 年份
        :param month: 月份，默认为None
        :param code: 股票代码，默认为None
        :return: 是否存在符合条件的数据

        匹配逻辑：
        - 如果month和code都提供：检查特定(year, month, code)是否存在
        - 如果只有month提供：检查该年份该月份是否有任何股票数据
        - 如果month和code都为None：检查该年份是否有任何数据
        """
        with self._lock:
            for key in self._cache.keys():
                cache_code, cache_year, cache_month = key

                # 必须匹配年份
                if cache_year != year:
                    continue

                # 如果指定了月份，必须匹配月份
                if month is not None and cache_month != month:
                    continue

                # 如果指定了代码，必须匹配代码
                if code is not None and cache_code != code:
                    continue

                # 找到匹配的数据
                return True

            return False
    
    @property
    def train_years_count(self) -> int:
        """
        获取年份数量
        :return: 年份数量
        """
        with self._lock:
            return len(set([year for _, year, _ in self._cache.keys()]))
    
    @property
    def current_train_year_month(self) -> Optional[Tuple[int, int]]:
        """当前训练进度，训练数据中最大(year,month)"""  
        with self._lock:
            if self._cache:
                return max([(year, month) for (_, year, month) in self._cache.keys()])
            return (self.get_start_year() - 1, 12) # 如果缓存为空，则返回开始年份-1和12月
        
    
    # ========== Meta数据接口 ==========
    def get_task_id(self) -> Optional[str]:
        """获取任务ID"""
        with self._meta_lock:
            return self._meta.get('task_id')
    
    def put_task_id(self, task_id: str):
        """设置任务ID"""
        with self._meta_lock:
            self._meta['task_id'] = task_id
    
    def get_start_year(self) -> Optional[int]:
        """获取开始年份"""
        with self._meta_lock:
            return self._meta.get('start_year')
    
    def put_start_year(self, year: int):
        """设置开始年份"""
        with self._meta_lock:
            self._meta['start_year'] = year
    
    def get_N(self) -> Optional[int]:
        """获取总股票数量"""
        with self._meta_lock:
            return self._meta.get('N')
    
    def put_N(self, n: int):
        """设置总股票数量"""
        with self._meta_lock:
            self._meta['N'] = n
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        with self._meta_lock:
            return self._meta.get('stock_list', [])
    
    def put_stock_list(self, stock_list: List[str]):
        """设置股票列表"""
        with self._meta_lock:
            self._meta['stock_list'] = stock_list
    
    def get_train_config(self) -> Optional[Dict]:
        """获取训练配置（仅随机部分，结构见本文件顶部【train_config = 随机】）"""
        with self._meta_lock:
            return self._meta.get('train_config')
    
    def put_train_config(self, config: Dict):
        """设置训练配置。若 config 含固定项（n/max_portfolios_num/env_config/performance_config），会拆出到 meta 顶层；仅随机部分写入 train_config。"""
        with self._meta_lock:
            config = dict(config)
            if 'n' in config:
                self._meta['n'] = config.pop('n')
            if 'max_portfolios_num' in config:
                self._meta['max_portfolios_num'] = config.pop('max_portfolios_num')
            self._meta['train_config'] = config
    
    def get_factors_list(self) -> List[str]:
        """获取因子列表"""
        with self._meta_lock:
            return self._meta.get('factors_list', [])
    
    def put_factors_list(self, factors_list: List[str]):
        """设置因子列表"""
        with self._meta_lock:
            self._meta['factors_list'] = factors_list
    
    def get_earliest_year_month(self) -> Optional[Tuple[int, int]]:
        """获取最早的年份和月份"""
        with self._meta_lock:
            return self._meta.get('earliest_year_month')
    
    def put_earliest_year_month(self, year, month):
        """设置最早的年份和月份"""
        with self._meta_lock:
            self._meta['earliest_year_month'] = (year, month)

    def get_n(self) -> Optional[int]:
        """获取一个组合中的证券数量（固定，meta 顶层。算上现金共 n+1 个）"""
        with self._meta_lock:
            return self._meta.get('n')

    def put_n(self, n: int):
        """设置一个组合中的证券数量（固定，meta 顶层）"""
        with self._meta_lock:
            self._meta['n'] = n

    def get_max_portfolios_num(self) -> Optional[int]:
        """获取可构建组合数上限（固定，meta 顶层）"""
        with self._meta_lock:
            return self._meta.get('max_portfolios_num')

    def put_max_portfolios_num(self, num: int):
        """设置可构建组合数上限（固定，meta 顶层）"""
        with self._meta_lock:
            self._meta['max_portfolios_num'] = num

    def get_env_config(self) -> Optional[Dict]:
        """获取环境配置（固定，meta 顶层。含 rf_end_year）"""
        with self._meta_lock:
            return self._meta.get('env_config')

    def put_env_config(self, config: Dict):
        """设置环境配置（固定，meta 顶层。含 rf_end_year）"""
        with self._meta_lock:
            self._meta['env_config'] = config

    def put_performance_config(self, config: Dict):
        """设置表现计算配置（固定，meta 顶层。含 risk_free_rate；vol/sharpe/mdd 窗口用 train_config.m）"""
        with self._meta_lock:
            self._meta['performance_config'] = config
    
    def get_performance_config(self) -> Optional[Dict]:
        """获取表现计算配置（固定，meta 顶层。含 risk_free_rate；vol/sharpe/mdd 窗口用 train_config.m）"""
        with self._meta_lock:
            return self._meta.get('performance_config')
    
    def get_end_year(self) -> Optional[int]:
        """获取结束年"""
        with self._meta_lock:
            return self._meta.get('end_year')
    
    def put_end_year(self, year: int):
        """设置结束年份（结束月份固定为 12，与 start_year 风格一致）"""
        with self._meta_lock:
            self._meta['end_year'] = year
            self._meta['end_month'] = 12
    
    def get_short_limit(self) -> Optional[float]:
        """获取短限制"""
        with self._meta_lock:
            return self._meta.get('short_limit')
    
    def put_short_limit(self, limit: float):
        """设置做空限制"""
        with self._meta_lock:
            self._meta['short_limit'] = limit

    def get_meta(self) -> Dict:
        """获取元数据（结构见本文件顶部：顶层固定 + train_config 随机）"""
        with self._meta_lock:
            return dict(self._meta)
    
    # =========== 记录数据接口（_record 独立字典） ============
    def get_record(self) -> Dict:
        """获取运行记录"""
        with self._record_lock:
            return dict(self._record)
    
    def put_record(self, key: str, value: Any):
        """设置运行记录中的某个字段"""
        with self._record_lock:
            self._record[key] = value
    
    def get_current_year_month(self) -> Optional[Tuple[int, int]]:
        """获取当前窗口 (year, month)"""
        with self._record_lock:
            return self._record.get('current_year_month')
    
    def put_current_year_month(self, year: int, month: int):
        """设置当前窗口 (year, month)"""
        with self._record_lock:
            self._record['current_year_month'] = (year, month)
    
    def get_running(self) -> bool:
        """获取运行状态"""
        with self._record_lock:
            return self._record.get('running', False)
    
    def put_running(self, running: bool):
        """设置运行状态"""
        with self._record_lock:
            self._record['running'] = running

    def get_pid(self) -> Optional[int]:
        """获取进程号"""
        with self._record_lock:
            return self._record.get('pid')
    
    def put_pid(self, pid: int):
        """设置进程号"""
        with self._record_lock:
            self._record['pid'] = pid

    def get_node_id(self) -> Optional[str]:
        """获取节点id"""
        with self._record_lock:
            return self._record.get('node_id')
    
    def put_node_id(self, node_id: str):
        """设置节点id"""
        with self._record_lock:
            self._record['node_id'] = node_id
# 全局实例  
DATA_CACHE_POOL = DataCachePool() 