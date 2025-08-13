"""
文档查重API服务主入口
使用模块化架构的新版本
"""

import os

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
