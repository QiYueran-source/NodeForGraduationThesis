"""
Redis连接管理
"""
# 库
import redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
import yaml
import dotenv
import os 
import time 

# 加载环境变量
dotenv.load_dotenv()

# 异常
from src.worker.data.redis.exception import (
    RedisException,
    RedisConnectionException,
    RedisConfigurationException
)

# 日志  
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix='[RedisConnection]')

# 可重试的异常类型（网络/超时类）
_RETRYABLE_EXCEPTIONS = (
    redis.ConnectionError,
    redis.TimeoutError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _retry_delay_for_attempt(config: dict, attempt: int) -> float:
    """根据 client_retry 配置计算第 attempt 次重试前的等待秒数（attempt 从 0 开始）。"""
    base = config.get('retry_delay', 1)
    if config.get('exponential_backoff', False):
        delay = base * (2 ** attempt)
        cap = config.get('max_delay', 30)
        return min(delay, cap)
    return float(base)


class RetryablePipeline:
    """对 pipeline.execute() 做重试的 pipeline 包装，配置来自 redis.yaml 的 client_retry。"""

    def __init__(self, pipe: redis.client.Pipeline, retry_config: dict):
        self._pipe = pipe
        self._retry_config = retry_config or {}

    def execute(self, raise_on_error: bool = True):
        cfg = self._retry_config
        if not cfg.get('enabled', True):
            return self._pipe.execute(raise_on_error=raise_on_error)

        max_retries = cfg.get('max_retries', 3)
        last_exc = None
        for attempt in range(max_retries):
            try:
                return self._pipe.execute(raise_on_error=raise_on_error)
            except _RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                if attempt < max_retries - 1:
                    delay = _retry_delay_for_attempt(cfg, attempt)
                    logger.warning(
                        f"pipeline.execute() 失败 (第 {attempt + 1}/{max_retries} 次): {e}，"
                        f"{delay}s 后重试"
                    )
                    time.sleep(delay)
        logger.error(f"pipeline.execute() 重试 {max_retries} 次后仍失败: {last_exc}")
        raise last_exc

    def __enter__(self):
        self._pipe.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self._pipe.__exit__(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._pipe, name)


class RetryableClient:
    """
    对 Redis 客户端的包装，对 get/set/incr/scan/pipeline 等常用方法做重试，
    其它方法透传到底层 client。重试配置从 redis.yaml 的 client_retry 段读取。
    """

    def __init__(self, client: redis.Redis):
        self._client = client
        self._retry_config = {}
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open('src/config/redis.yaml', 'r', encoding='utf-8') as f:
                self._retry_config = yaml.safe_load(f).get('client_retry', {})
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise

    def _retry(self, fn, *args, **kwargs):
        cfg = self._retry_config
        if not cfg.get('enabled', True):
            return fn(*args, **kwargs)

        max_retries = cfg.get('max_retries', 3)
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except _RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                if attempt < max_retries - 1:
                    delay = _retry_delay_for_attempt(cfg, attempt)
                    logger.warning(
                        f"Redis 操作失败 (第 {attempt + 1}/{max_retries} 次): {e}，"
                        f"{delay}s 后重试"
                    )
                    time.sleep(delay)
        logger.error(f"Redis 操作重试 {max_retries} 次后仍失败: {last_exc}")
        raise last_exc

    def get(self, name, *args, **kwargs):
        return self._retry(self._client.get, name, *args, **kwargs)

    def set(self, name, value, *args, **kwargs):
        return self._retry(self._client.set, name, value, *args, **kwargs)

    def incr(self, name, amount=1, *args, **kwargs):
        return self._retry(self._client.incr, name, amount, *args, **kwargs)

    def scan(self, cursor=0, match=None, count=None, *args, **kwargs):
        return self._retry(
            self._client.scan, cursor, match=match, count=count, *args, **kwargs
        )

    def ping(self, *args, **kwargs):
        return self._retry(self._client.ping, *args, **kwargs)

    def pipeline(self, transaction=True, shard_hint=None):
        pipe = self._client.pipeline(transaction=transaction, shard_hint=shard_hint)
        return RetryablePipeline(pipe, self._retry_config)

    def close(self, *args, **kwargs):
        return self._client.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


# redis连接管理  
class RedisConnector:
    def __init__(self,create_client:bool = True):
        # 基本配置  
        self._client = None
        self._is_connected = False
        self._last_health_check = 0  
        self._config = {}

        # 加载配置
        self._load_config()

        # 建立连接 
        if create_client:
            self._create_client()
    
    def _load_config(self):
        """加载Redis配置"""
        password = os.getenv('REDIS_PWD', 'failed')
        if password == 'failed':
            logger.error(f"Redis密码读取失败")
            raise RedisConfigurationException(f"Redis密码读取失败")

        try:
            with open('src/config/redis.yaml', 'r', encoding = 'utf-8') as f:
                self._config = yaml.safe_load(f)
                if 'parameters' in self._config:
                    self._config['parameters']['password'] = password
                else:
                    raise 

        except Exception as e:
            logger.error(f"Redis配置读取失败: {e}")
            raise RedisConfigurationException(f"Redis配置读取失败: {e}")
    
    def _create_client(self):
        """初始化Redis客户端"""
        # 初始化参数
        param = self._config.get('parameters', {})

        # 重试配置
        is_retry = self._config.get('retry',{}).get('retry_on_timeout', False)
        if is_retry:
            retry_param = self._config.get('retry',{})
            retry_delay = retry_param.get('retry_delay', 1)
            retry_cap_multiplier = retry_param.get('cap_multiplier', 10)
            retry_max_retries = retry_param.get('max_retries', 3)
            retry_config = Retry(
                backoff=ExponentialBackoff(
                    cap=retry_delay * retry_cap_multiplier,
                    base=retry_delay
                ),
                retries=retry_max_retries
            )

            param = {
                **param,
                'retry': retry_config
            }
        
        try:
            self._client = redis.Redis(**param)
            logger.debug(f"Redis连接创建成功: {self._client}")
        except Exception as e:
            logger.error(f"Redis连接创建失败: {e}")
            raise RedisConnectionException(f"Redis连接创建失败: {e}")

    def _health_check(self) -> bool:
        """检查Redis连接是否健康"""
        try:
            if self._client:
                self._client.ping()
                self._last_health_check = time.time()
                self._is_connected = True
                logger.debug(f"Redis连接健康检查成功")
                return True
        except Exception as e:
            logger.warning(f"Redis健康检查失败: {e}")
            self._is_connected = False
            return False
        return False
    
    def _reconnect(self) -> None:
        """重新连接Redis"""
        logger.info("开始重新连接Redis...")
        try:
            # 清理旧连接
            if self._client:
                self._client.close()
            
            # 重新创建连接
            self._create_client()
            logger.info("Redis重连成功")
            
        except Exception as e:
            logger.error(f"Redis重连失败: {e}")
            raise RedisConnectionException(f"重连失败: {e}")
    
    def get_client(self):
        """获取Redis客户端 (核心接口)，返回带重试的包装 client。"""
        # 检查连接状态
        if self._client is None or not self._is_connected:
            self._create_client()  # 创建连接

        # 定期健康检查
        current_time = time.time()
        health_check_interval = self._config.get('monitoring', {}).get('health_check_interval', 30)
        if current_time - self._last_health_check > health_check_interval:
            if not self._health_check():
                logger.warning("Redis健康检查失败，尝试重连...")
                self._reconnect()

        return RetryableClient(self._client)

    def close(self) -> None:
        """关闭Redis连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._is_connected = False
            logger.info("Redis连接已关闭")
        else:
            logger.warning("Redis连接已关闭")

REDIS_CONNECTOR = RedisConnector()