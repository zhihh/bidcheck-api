"""
FastAPI应用定义
包含所有API路由和中间件配置
"""

import logging
import json
from datetime import datetime
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from ..models.api_models import DocumentInput, ApiResponse
from .service import DocumentDeduplicationService
from ..config.config import Config

logger = logging.getLogger(__name__)

# 创建配置实例
config = Config()

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

# 请求日志中间件 - 在所有处理之前记录
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """请求日志中间件 - 无条件捕获所有请求"""
    
    # 无条件记录所有请求信息
    logger.info("🚨" + "=" * 80)
    logger.info("� [RAW REQUEST] 收到原始请求")
    logger.info("🚨" + "=" * 80)
    logger.info(f"📍 URL: {request.url}")
    logger.info(f"📝 方法: {request.method}")
    logger.info(f"🌐 客户端: {request.client.host if request.client else 'Unknown'}:{request.client.port if request.client else 'Unknown'}")
    logger.info(f"� 所有请求头:")
    for name, value in request.headers.items():
        logger.info(f"    {name}: {value}")
    
    # 读取请求体
    body = await request.body()
    logger.info(f"📏 请求体字节长度: {len(body)}")
    
    if body:
        logger.info("📄 原始请求体 (字节):")
        logger.info(f"    {body}")
        
        try:
            # 尝试UTF-8解码
            body_str = body.decode('utf-8')
            logger.info("📄 请求体 (UTF-8字符串):")
            logger.info(f"    {repr(body_str)}")
            logger.info("📄 请求体 (显示内容):")
            logger.info(body_str)
            
            # 尝试JSON解析
            try:
                parsed = json.loads(body_str)
                logger.info("✅ JSON解析成功:")
                logger.info(json.dumps(parsed, ensure_ascii=False, indent=2))
                
                # 详细分析数据结构
                if isinstance(parsed, list):
                    logger.info(f"📊 数据类型: 数组，长度: {len(parsed)}")
                    for i, item in enumerate(parsed[:3]):  # 只显示前3个
                        logger.info(f"🔸 元素[{i}]: {type(item)} = {item}")
                        if isinstance(item, dict):
                            logger.info(f"    键: {list(item.keys())}")
                elif isinstance(parsed, dict):
                    logger.info(f"📊 数据类型: 对象")
                    logger.info(f"🔑 对象键: {list(parsed.keys())}")
                    for key, value in list(parsed.items())[:5]:  # 只显示前5个键值对
                        logger.info(f"    {key}: {type(value)} = {value}")
                        
            except json.JSONDecodeError as je:
                logger.error(f"❌ JSON解析失败: {je}")
                logger.error(f"    位置: 行{je.lineno}, 列{je.colno}")
                logger.error(f"    消息: {je.msg}")
                
        except UnicodeDecodeError as ue:
            logger.error(f"❌ UTF-8解码失败: {ue}")
            logger.info(f"� 尝试其他编码...")
            try:
                body_str = body.decode('latin1')
                logger.info(f"📄 Latin1解码: {body_str}")
            except:
                logger.error("❌ 所有编码尝试都失败")
                
        except Exception as e:
            logger.error(f"❌ 请求体处理异常: {e}")
    else:
        logger.info("📄 请求体为空")
    
    logger.info("🚨" + "=" * 80)
    
    # 重新构造请求对象（因为body只能读一次）
    async def receive():
        return {"type": "http.request", "body": body}
    
    request._receive = receive
    
    # 继续处理请求
    try:
        response = await call_next(request)
        logger.info(f"📤 [RESPONSE] 状态码: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"❌ [MIDDLEWARE ERROR] 处理请求时异常: {e}")
        logger.error(f"    异常类型: {type(e).__name__}")
        raise

# 422验证错误处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422验证错误处理器 - 捕获数据验证失败"""
    logger.error("🚨 " + "=" * 60)
    logger.error("🚨 422 数据验证错误")
    logger.error("🚨 " + "=" * 60)
    logger.error(f"📍 请求路径: {request.url.path}")
    logger.error(f"📝 请求方法: {request.method}")
    logger.error(f"❌ 验证错误详情:")
    
    for error in exc.errors():
        logger.error(f"  🔸 位置: {' -> '.join(str(loc) for loc in error['loc'])}")
        logger.error(f"  🔸 错误: {error['msg']}")
        logger.error(f"  🔸 类型: {error['type']}")
        if 'input' in error:
            logger.error(f"  🔸 输入值: {error['input']}")
    
    # 尝试再次读取请求体
    try:
        body = await request.body()
        if body:
            body_str = body.decode('utf-8')
            logger.error(f"📄 失败的请求体: {body_str}")
            
            # 尝试解析JSON查看结构
            try:
                parsed = json.loads(body_str)
                logger.error(f"📊 解析后的JSON结构:")
                logger.error(json.dumps(parsed, ensure_ascii=False, indent=2))
            except:
                pass
    except Exception as e:
        logger.error(f"❌ 无法读取请求体: {e}")
    
    logger.error("🚨 " + "=" * 60)
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求数据验证失败",
            "message": "数据格式不符合API要求",
            "details": exc.errors(),
            "expected_format": {
                "example": [
                    {
                        "documentId": 1,
                        "page": 1,
                        "content": "文档内容"
                    }
                ]
            }
        }
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
async def analyze_documents(request: Request, documents: List[DocumentInput]):
    """
    分析文档重复内容 - 异步并发版本
    
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
    
    # 获取原始请求体用于调试（可通过环境变量控制）
    if config.debug_request_body:
        try:
            request_body = await request.body()
            request_text = request_body.decode('utf-8')
            logger.info(f"🔍 [DEBUG] 收到请求，原始请求体: {request_text}")
            logger.info(f"📊 [DEBUG] 请求头: {dict(request.headers)}")
            logger.info(f"🎯 [DEBUG] 解析后的文档数量: {len(documents)}")
            
            # 记录解析后的文档结构
            for i, doc in enumerate(documents[:3]):  # 只记录前3个文档避免日志过长
                logger.info(f"📄 [DEBUG] 文档 {i+1}: documentId={doc.documentId}, page={doc.page}, content长度={len(doc.content)}")
                logger.info(f"📄 [DEBUG] 文档 {i+1} 内容前100字符: {doc.content[:100]}...")
                
        except Exception as e:
            logger.error(f"❌ [DEBUG] 获取请求体失败: {e}")
    else:
        logger.info(f"📨 收到分析请求，文档数量: {len(documents)}")
        # 重新读取请求体用于重新构造request对象（因为body只能读取一次）
        # 这里我们直接使用已解析的documents
    
    try:
        # 验证输入
        if not documents:
            logger.warning("⚠️ 输入文档为空")
            raise HTTPException(status_code=400, detail="输入文档不能为空")
        
        logger.info(f"✅ 输入验证通过，文档数量: {len(documents)}")
        
        # 转换为字典格式
        json_input = [
            {
                "documentId": doc.documentId,
                "page": doc.page,
                "content": doc.content
            }
            for doc in documents
        ]
        
        logger.info(f"🔄 转换为内部格式完成")
        
        # 执行异步分析
        duplicate_results = await deduplication_service.analyze_documents(json_input)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ 分析完成，发现 {len(duplicate_results)} 对重复内容，耗时 {processing_time:.2f}秒")
        
        return ApiResponse(
            success=True,
            message=f"分析完成，发现 {len(duplicate_results)} 对重复内容",
            data=duplicate_results,
            total_count=len(duplicate_results),
            processing_time=processing_time
        )
        
    except HTTPException as he:
        logger.error(f"❌ HTTP异常: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"❌ API调用失败: {e}")
        logger.error(f"❌ 异常类型: {type(e).__name__}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ApiResponse(
            success=False,
            message=f"分析失败: {str(e)}",
            data=None,
            total_count=0,
            processing_time=processing_time
        )


@app.post("/api/v2/debug/toggle")
async def toggle_debug_mode(enable: Optional[bool] = None):
    """动态切换调试模式
    
    Args:
        enable: true开启调试，false关闭调试，不传参数则切换当前状态
    """
    import os
    
    current_state = config.debug_request_body
    
    if enable is None:
        # 切换当前状态
        new_state = not current_state
    else:
        new_state = enable
    
    # 动态修改环境变量
    os.environ["DEBUG_REQUEST_BODY"] = "true" if new_state else "false"
    
    logger.info(f"🔧 调试模式已{'开启' if new_state else '关闭'} (原状态: {'开启' if current_state else '关闭'})")
    
    return {
        "status": "success",
        "message": f"调试模式已{'开启' if new_state else '关闭'}",
        "previous_state": current_state,
        "current_state": new_state,
        "usage": {
            "开启调试": "POST /api/v2/debug/toggle?enable=true",
            "关闭调试": "POST /api/v2/debug/toggle?enable=false", 
            "切换状态": "POST /api/v2/debug/toggle"
        }
    }


@app.get("/api/v2/debug/status")
async def get_debug_status():
    """获取当前调试状态"""
    return {
        "debug_request_body": config.debug_request_body,
        "message": f"请求体调试日志当前{'已开启' if config.debug_request_body else '已关闭'}",
        "controls": {
            "开启调试": "POST /api/v2/debug/toggle?enable=true",
            "关闭调试": "POST /api/v2/debug/toggle?enable=false",
            "切换状态": "POST /api/v2/debug/toggle"
        }
    }


@app.post("/api/v2/debug")
async def debug_request(request: Request):
    """调试端点 - 记录原始请求体用于调试422错误"""
    try:
        # 获取原始请求体
        request_body = await request.body()
        request_text = request_body.decode('utf-8')
        
        logger.info("=" * 60)
        logger.info("🔍 调试请求信息")
        logger.info("=" * 60)
        logger.info(f"📊 请求头: {dict(request.headers)}")
        logger.info(f"🎯 Content-Type: {request.headers.get('content-type', 'Not Set')}")
        logger.info(f"📏 请求体长度: {len(request_body)} 字节")
        logger.info(f"📄 原始请求体:")
        logger.info(request_text)
        logger.info("=" * 60)
        
        # 尝试解析JSON
        try:
            parsed_json = json.loads(request_text)
            logger.info(f"✅ JSON解析成功，类型: {type(parsed_json)}")
            if isinstance(parsed_json, list):
                logger.info(f"📊 数组长度: {len(parsed_json)}")
                if parsed_json:
                    logger.info(f"🎯 第一个元素: {parsed_json[0]}")
                    logger.info(f"🔑 第一个元素的键: {list(parsed_json[0].keys()) if isinstance(parsed_json[0], dict) else 'Not a dict'}")
            elif isinstance(parsed_json, dict):
                logger.info(f"🔑 对象的键: {list(parsed_json.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析失败: {e}")
        
        return {
            "status": "debug_success",
            "message": "请求信息已记录到日志",
            "content_length": len(request_body),
            "content_type": request.headers.get('content-type', 'Not Set')
        }
        
    except Exception as e:
        logger.error(f"❌ 调试端点异常: {e}")
        return {
            "status": "debug_error",
            "message": f"调试失败: {str(e)}"
        }


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
async def test_with_sample_data(request: Request):
    """使用示例数据测试API - 异步版本"""
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
    
    # 调用异步分析接口
    return await analyze_documents(request, documents)
