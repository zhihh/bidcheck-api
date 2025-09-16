"""
文档查重API服务主入口 - 高并发生产版本
使用模块化架构的新版本，支持多工作进程和异步处理
"""

import os
import multiprocessing

import uvicorn
from dotenv import load_dotenv

from src.config.config import Config

# 检查 .env 文件是否存在
if os.path.exists(".env"):
    load_dotenv(".env")

# 初始化配置
config = Config()

from src.api.app import app

if __name__ == "__main__":
    # 获取CPU核心数，用于确定工作进程数
    workers = int(os.environ.get("WORKERS", min(multiprocessing.cpu_count(), 4)))
    
    # 生产环境配置
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=workers,  # 多工作进程
        log_level="info", 
        access_log=True,
        reload=False,  # 生产环境关闭自动重载
        loop="uvloop",  # 使用高性能事件循环
        http="httptools"  # 使用高性能HTTP解析器
    )
