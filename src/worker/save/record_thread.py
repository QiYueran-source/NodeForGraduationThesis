"""
record 线程
定时将状态写入 record.json，供 tcp_reciver 读取查询
"""
# 库

# 组件
from src.worker.save import SAVER

# 日志
from src.utils.logger import get_module_logger
logger = get_module_logger(__name__, prefix="RecordThread")


def record_thread_start():
    """启动 record 线程"""
    logger.info("record 线程开始执行")
    SAVER.start_record_thread()
    logger.debug("record 线程已启动，将按间隔写入 record.json 供 tcp 查询")


def record_thread_stop():
    """停止 record 线程"""
    logger.info("record 线程停止执行")
    SAVER.stop_record_thread()
    logger.debug("record 线程已停止")
