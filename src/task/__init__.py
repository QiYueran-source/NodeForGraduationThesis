"""
任务  
"""
# 库
from pathlib import Path
import datetime as dt
from typing import TypedDict

class Status:
    """
    任务执行状态  
    """

class Task(TypedDict):
    """
    任务类，由tcp_reciver维护，记录本次任务的内容
    """
    task_id:str # 任务id  
    exe_time:dt.datetime # 开始执行时间
    agent_config:dict # agent配置  
    status:Status # 状态  
    base_path:Path = Path(f'/Node/data/{task_id}')# 结果保存路径  