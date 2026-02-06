# 库
import json
from typing import List, Optional, Dict, Tuple, Any

# 自定义组件 
from src.worker.data.redis import REDIS_CONNECTOR,REDIS_PREFIX_MANAGER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[DataLoaderFetch]')

# 数据加载层
class DataLoader:
    def __init__(self):
        self.client = REDIS_CONNECTOR.get_client()
        self.redis_prefix_manager = REDIS_PREFIX_MANAGER

    def _parse_json(self, raw: Any) -> Optional[Any]:
        """将 Redis 返回的原始值（bytes/str）解析为 Python 对象，失败返回 None 并打日志。"""
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"JSON 解析失败: {e}")
            return None

    def fetch_data(self, 
        year: int, 
        month: Optional[int] = None, 
        code_list: Optional[List[str]] = None 
    ) -> Dict[Tuple[int, str], Any]:
        """
        从redis中获取数据
        :param year: 年份
        :param month: 月份，如果提供则只加载该月份，否则加载所有月份(1-12)
        :param code_list: 股票代码列表，如果提供则只加载这些代码，否则加载所有代码
        :return: 字典，键为(month, code)元组，值为数据
        """
        logger.debug(f"fetch_data 开始: year={year}, month={month}, code_list={f'指定{len(code_list)}只' if code_list else '全部'}")
        # 确定要加载的月份列表
        if month is not None:
            months = [month]
        else:
            months = list(range(1, 13))  # 1-12月
        
        # 确定要加载的代码列表
        if code_list is not None:
            codes = code_list
            logger.debug(f"使用指定代码列表: {len(codes)} 只")
        else:
            # 从Redis中扫描获取所有代码
            codes = self._get_all_codes(year, months)
            logger.debug(f"Redis 扫描得到代码数: year={year}, codes={len(codes)}")
        
        # 使用Redis流水线批量加载数据
        result = {}
        data_keys = []
        key_mapping = []

        # 第一步：收集所有需要获取的键
        for m in months:
            for code in codes:
                slice_key = self.redis_prefix_manager.build_train_slice_key(year, m, code)
                data_keys.append(slice_key)
                key_mapping.append((m, code))

        # 第二步：使用pipeline批量获取所有数据
        try:
            with self.client.pipeline() as pipe:
                # 批量添加GET命令
                for key in data_keys:
                    pipe.get(key)

                # 执行批量获取
                responses = pipe.execute()

            # 第三步：处理响应（解析 JSON）并批量增加计数器
            counters_to_incr = []
            for i, response in enumerate(responses):
                if response is not None:  # 数据存在
                    parsed = self._parse_json(response)
                    if parsed is None:
                        continue
                    month, code = key_mapping[i]
                    result[(month, code)] = parsed

                    # 收集需要增加的计数器键
                    counter_key = self.redis_prefix_manager.build_counter_key(year, month, code)
                    counters_to_incr.append(counter_key)

            # 第四步：批量增加计数器
            if counters_to_incr:
                with self.client.pipeline() as pipe:
                    for counter_key in counters_to_incr:
                        pipe.incr(counter_key)

                    counter_responses = pipe.execute()

                logger.info(f"批量加载数据: year={year}, 请求={len(data_keys)}, 成功={len(result)}, 计数器更新={len(counters_to_incr)}")

        except Exception as e:
            logger.warning(f"批量加载数据失败 year={year}: {e}")
            # 如果批量操作失败，回退到逐个获取（保证可用性）
            logger.info("回退到逐个加载模式...")
            result = self._fetch_data_fallback(year, months, codes)

        logger.debug(f"fetch_data 结束: year={year}, 返回条数={len(result)}")
        return result
    
    def _get_all_codes(self, year: int, months: List[int]) -> List[str]:
        """
        从Redis中扫描获取指定年份和月份的所有代码
        :param year: 年份
        :param months: 月份列表
        :return: 代码列表
        """
        codes_set = set()
        train_prefix = self.redis_prefix_manager.train_prefix
        
        # 对每个月份扫描键
        for month in months:
            month_str = f"{month:02d}"
            # 构建扫描模式: gt:data:train:{year}:{month}:*
            pattern = f"{train_prefix}:{year}:{month_str}:*"
            
            # 使用scan迭代所有匹配的键
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    # 从键中提取代码: gt:data:train:{year}:{month}:{code}
                    parts = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = parts.split(':')
                    if len(parts) >= 5:
                        code = parts[-1]  # 最后一部分是代码
                        codes_set.add(code)
                
                if cursor == 0:
                    break
            logger.debug(f"_get_all_codes 月份 {month_str}: 扫描到 {len(codes_set)} 个代码")
        
        out = sorted(list(codes_set))
        logger.debug(f"_get_all_codes 合计: year={year}, 代码数={len(out)}")
        return out

    def _fetch_data_fallback(self, year: int, months: List[int], codes: List[str]) -> Dict[Tuple[int, str], Any]:
        """
        回退方法：逐个加载数据（当批量操作失败时使用）
        :param year: 年份
        :param months: 月份列表
        :param codes: 代码列表
        :return: 数据字典
        """
        result = {}
        for m in months:
            for code in codes:
                try:
                    # 构建键
                    slice_key = self.redis_prefix_manager.build_train_slice_key(year, m, code)
                    counter_key = self.redis_prefix_manager.build_counter_key(year, m, code)

                    # 获取数据并解析 JSON
                    data = self.client.get(slice_key)
                    if data is not None:
                        parsed = self._parse_json(data)
                        if parsed is not None:
                            result[(m, code)] = parsed
                            self.client.incr(counter_key)
                except Exception as e:
                    logger.warning(f"加载数据失败(回退模式) year={year}, month={m}, code={code}: {e}")
                    continue

        logger.info(f"回退模式加载完成: year={year}, 成功加载={len(result)}")
        logger.debug(f"回退模式 fetch_data 结束: year={year}")
        return result

DATA_LOADER = DataLoader()