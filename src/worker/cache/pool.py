"""
数据缓存池
线程安全的数据缓存，用于Worker进程内线程间通信  
包括训练数据、元数据和记录数据  
train: 由主机提供  
    - (code, year, month): [[因子],收益]   

meta：由主机提供   
- task_id: 任务id    
- start_year: 开始年份  
- N: 总股票数量    
- stock_list: 股票列表   
- factors_list: 因子列表（避免麻烦，直接保存本地）   
- earliest_year_month: 最早的年份和月份,(year, month)  
- train_config: 训练配置   
    - n: 一个组合中的证券数量（算上现金，共n+1个证券）  
    - max_portfolios_num: 对于总共n个证券，最多可以构建C(N,n)个组合,太大，所以设置最大组合数量    
    - m: 回看的期数      
    - mask: 因子掩码，1表示看，0表示不看    

record: 由节点维护  
- running: 是否正在运行    
- current_year_month: 当前窗口(year,month)   
 
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
        
        # 结构化meta数据
        self._meta: Dict[str, Any] = {
            'task_id': None,
            'start_year': None,
            'N': None,
            'stock_list': [],
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
            'train_config': {
                'n': None,
                'max_portfolios_num': None,
                'm': None,
                'mask': None
            },
            'record': {
                'running': False,
                'current_year_month': None
            }
        } 
        
        
        # 线程锁，保护缓存操作
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        
        logger.info("数据缓存池已创建")
    
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
                logger.warning("数据已存在，将被覆盖: %s, %s, %s", code, year, month)
            self._cache[key] = data
            logger.debug("数据已放入缓存: %s, %s, %s", code, year, month)
    
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
                    logger.warning("批量插入项格式错误，跳过: %s", item)
                    continue
                
                key = (code, year, month)
                if key in self._cache:
                    logger.debug("批量插入：数据已存在，将被覆盖: %s, %s, %s", code, year, month)
                
                self._cache[key] = data
                count += 1
            
            logger.info("批量插入完成，共插入 %d 条数据", count)
    
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
                logger.debug("数据已从缓存取出并删除: %s, %s, %s", code, year, month)
                return data
            else:
                logger.debug("缓存中不存在: %s, %s, %s", code, year, month)
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
    
    def count_years_train(self) -> int:
        """
        获取年份数量
        :return: 年份数量
        """
        with self._lock:
            return len(set([year for _, year, _ in self._cache.keys()]))
    
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
        """获取训练配置"""
        with self._meta_lock:
            return self._meta.get('train_config')
    
    def put_train_config(self, config: Dict):
        """设置训练配置"""
        with self._meta_lock:
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

    # =========== 记录数据接口 ============
    def get_record(self) -> Dict:
        """获取运行记录"""
        with self._meta_lock:
            record = self._meta.get('record', {})
            if not isinstance(record, dict):
                self._meta['record'] = {'running': False, 'current_year_month': None}
                return self._meta['record']
            return record
    
    def put_record(self, key: str, value: Any):
        """设置运行记录中的某个字段"""
        with self._meta_lock:
            if 'record' not in self._meta or not isinstance(self._meta['record'], dict):
                self._meta['record'] = {'running': False, 'current_year_month': None}
            self._meta['record'][key] = value
    
    def get_current_year_month(self) -> Optional[Tuple[int, int]]:
        """获取当前窗口 (year, month)"""
        record = self.get_record()
        return record.get('current_year_month') if isinstance(record, dict) else None
    
    def put_current_year_month(self, year: int, month: int):
        """设置当前窗口 (year, month)"""
        self.put_record('current_year_month', (year, month))
    
    def get_running(self) -> bool:
        """获取运行状态"""
        record = self.get_record()
        return record.get('running', False) if isinstance(record, dict) else False
    
    def put_running(self, running: bool):
        """设置运行状态"""
        self.put_record('running', running)
    
# 全局实例  
DATA_CACHE_POOL = DataCachePool() 