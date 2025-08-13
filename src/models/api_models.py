"""
API数据模型
定义API请求和响应的数据结构
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """输入文档数据结构"""
    documentId: int = Field(description="文档ID")
    page: int = Field(description="页码")
    content: str = Field(description="文档内容")


class DuplicateOutput(BaseModel):
    """输出重复检测结果"""
    documentId1: int = Field(description="第一个文档ID")
    page1: int = Field(description="第一个文档页码")
    chunkId1: int = Field(description="第一个文档片段ID")
    content1: str = Field(description="第一个文档片段内容")
    prefix1: str = Field(description="第一个文档片段前缀预览")
    suffix1: str = Field(description="第一个文档片段后缀预览")
    documentId2: int = Field(description="第二个文档ID")
    page2: int = Field(description="第二个文档页码")
    chunkId2: int = Field(description="第二个文档片段ID")
    content2: str = Field(description="第二个文档片段内容")
    prefix2: str = Field(description="第二个文档片段前缀预览")
    suffix2: str = Field(description="第二个文档片段后缀预览")
    reason: str = Field(description="大模型判断的原因")
    score: float = Field(description="重复程度得分(0-1)")


class ApiResponse(BaseModel):
    """API响应格式"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="响应消息")
    data: Optional[List[DuplicateOutput]] = Field(description="重复检测结果")
    total_count: int = Field(description="重复对总数", default=0)
    processing_time: Optional[float] = Field(description="处理时间(秒)")
