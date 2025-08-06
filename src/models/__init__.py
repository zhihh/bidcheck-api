"""
数据模型模块
定义系统中使用的所有数据结构
"""

from .api_models import DocumentInput, DuplicateOutput, ApiResponse
from .data_models import TextSegment, DocumentData

__all__ = [
    'DocumentInput',
    'DuplicateOutput', 
    'ApiResponse',
    'TextSegment',
    'DocumentData'
]
