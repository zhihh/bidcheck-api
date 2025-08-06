"""
内部数据模型
定义系统内部使用的数据结构
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TextSegment:
    """文本片段数据结构"""
    id: str
    content: str
    document_id: int
    page: int
    chunk_id: int
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None


@dataclass
class DocumentData:
    """完整文档数据结构"""
    document_id: int
    content: str  # 合并所有页面的内容
    pages: Dict[int, str]  # 页面ID到内容的映射
