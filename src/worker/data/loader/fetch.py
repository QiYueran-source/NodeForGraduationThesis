# 库
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

# 自定义组件
from src.worker.data.redis import REDIS_CONNECTOR, REDIS_PREFIX_MANAGER
from src.worker.cache.pool import DATA_CACHE_POOL

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[DataLoaderFetch]')

# Lua 脚本：原子地将 node_id 追加到 counter key 的 JSON 数组（无则追加）
_SCRIPT_DIR = Path(__file__).resolve().parent
_APPEND_NODE_SCRIPT = (_SCRIPT_DIR / "append_node_to_counter.lua").read_text(encoding="utf-8")


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
        f"""
        从redis中获取数据
        :param year: 年份
        :param month: 月份，如果提供则只加载该月份，否则加载所有月份(1-12)
        :param code_list: 股票代码列表，如果提供则只加载这些代码，否则加载所有代码
        :return: 字典，键为(month, code)元组，值为数据  
        如果返回空字典，则说明没有数据，需要重新获取  
        如果返回None，则说明出现异常 
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

            # 第三步：处理响应（解析 JSON）并收集需更新节点列表的 counter 键
            counters_to_append = []
            for i, response in enumerate(responses):
                if response is not None:  # 数据存在
                    parsed = self._parse_json(response)
                    if parsed is None:
                        continue
                    month, code = key_mapping[i]
                    result[(month, code)] = parsed

                    counter_key = self.redis_prefix_manager.build_counter_key(year, month, code)
                    counters_to_append.append(counter_key)

            # 第四步：原子地将本节点 node_id 追加到各 counter 的 JSON 节点列表（Lua 脚本）
            node_id = DATA_CACHE_POOL.get_node_id()
            if counters_to_append and node_id:
                with self.client.pipeline() as pipe:
                    for counter_key in counters_to_append:
                        pipe.eval(_APPEND_NODE_SCRIPT, 1, counter_key, node_id)
                    pipe.execute()
                logger.info(f"批量加载数据: year={year}, 请求={len(data_keys)}, 成功={len(result)}, 节点列表更新={len(counters_to_append)}")
            elif counters_to_append and not node_id:
                logger.warning("node_id 为空，跳过 counter 节点列表更新")
            # 有数据则报成功，无数据则报警告
            if len(result) > 0:
                print(f"成功加载数据: year={year}, 请求={len(data_keys)}, 成功={len(result)}")
            else:
                logger.warning(f"未加载到数据: year={year}, 准备重试")
        except Exception as e:
            logger.warning(f"批量加载数据失败 year={year}: {e}，返回None")
            return None  

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

DATA_LOADER = DataLoader()