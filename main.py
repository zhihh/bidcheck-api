"""
文档查重API服务主入口
使用模块化架构的新版本
"""

import uvicorn
from src.config.config import Config
# 初始化配置
config = Config()

from src.api.app import app



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
