"""
FastAPI应用定义
包含所有API路由和中间件配置
"""

import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..models.api_models import DocumentInput, ApiResponse
from .service import DocumentDeduplicationService

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="文档查重API服务",
    description="基于深度学习和大模型的文档查重系统",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
deduplication_service = DocumentDeduplicationService()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "文档查重API服务",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v2/analyze", response_model=ApiResponse)
async def analyze_documents(documents: List[DocumentInput]):
    """
    分析文档重复内容
    
    输入格式:
    [
        {
            "documentId": 1,
            "page": 1,
            "content": "文档内容"
        }
    ]
    
    输出格式包含重复内容对的详细信息
    """
    start_time = datetime.now()
    
    try:
        # 验证输入
        if not documents:
            raise HTTPException(status_code=400, detail="输入文档不能为空")
        
        # 转换为字典格式
        json_input = [
            {
                "documentId": doc.documentId,
                "page": doc.page,
                "content": doc.content
            }
            for doc in documents
        ]
        
        # 执行分析
        duplicate_results = deduplication_service.analyze_documents(json_input)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ApiResponse(
            success=True,
            message=f"分析完成，发现 {len(duplicate_results)} 对重复内容",
            data=duplicate_results,
            total_count=len(duplicate_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ApiResponse(
            success=False,
            message=f"分析失败: {str(e)}",
            data=None,
            total_count=0,
            processing_time=processing_time
        )


@app.get("/api/v2/status")
async def get_status():
    """获取服务状态"""
    return {
        "service": "Document Deduplication API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }


@app.post("/api/v2/test")
async def test_with_sample_data():
    """使用示例数据测试API"""
    # 生成测试数据
    test_data = [
        {
            "documentId": 1,
            "page": 1,
            "content": "人工智能是计算机科学的一个分支，它企图了解智能的实质。\n机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。\n深度学习是机器学习的一个分支，使用神经网络进行模式识别。"
        },
        {
            "documentId": 2,
            "page": 1,
            "content": "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。\n通过算法和统计模型，机器可以在没有明确编程的情况下执行任务。\n深度学习属于机器学习的一个分支，它使用神经网络来识别模式。"
        },
        {
            "documentId": 3,
            "page": 1,
            "content": "自然语言处理是人工智能的重要应用领域之一。\n语音识别技术已经广泛应用于智能助手中。\n推荐系统利用机器学习算法为用户提供个性化内容。"
        }
    ]
    
    # 转换为DocumentInput对象
    documents = [DocumentInput(**item) for item in test_data]
    
    # 调用分析接口
    return await analyze_documents(documents)
