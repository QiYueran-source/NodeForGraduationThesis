"""
Redis模块，用于管理Redis连接  
"""
from .connection import RedisConnector
from .keys import RedisPrefixManager

# 异常
from .exception import (
    RedisException, 
    RedisConnectionException, 
    RedisConfigurationException
)

# 全局连接管理器
REDIS_CONNECTOR = RedisConnector() 
REDIS_PREFIX_MANAGER = RedisPrefixManager()

__all__ = [
    'RedisConnector',
    'RedisException',
    'RedisConnectionException',
    'RedisConfigurationException',
    'REDIS_CONNECTOR',
    'REDIS_PREFIX_MANAGER'
]