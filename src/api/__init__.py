"""
API应用模块
包含FastAPI应用和路由定义
"""

from .app import app
from .service import DocumentDeduplicationService

__all__ = [
    'app',
    'DocumentDeduplicationService'
]
