"""
系统配置管理
"""

import os
import logging


class Config:
    """系统配置类"""
    
    def __init__(self):
        self._setup_environment()
        self._setup_logging()
    
    def _setup_environment(self):
        """设置环境变量"""
        # OpenAI 配置
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 模型配置
        os.environ["LLM_MODEL_NAME"] = os.getenv("LLM_MODEL_NAME", "qwen-turbo")
        os.environ["EMBEDDING_MODEL_NAME"] = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
        
        # LangSmith 配置
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = "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"
        os.environ["LANGSMITH_PROJECT"] = "BidCheck"
        
        # 调试配置
        os.environ["DEBUG_REQUEST_BODY"] = os.getenv("DEBUG_REQUEST_BODY", "false")
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(level=logging.INFO)
    
    @property
    def openai_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")
    
    @property
    def openai_base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "")
    
    @property
    def llm_model_name(self) -> str:
        return os.environ.get("LLM_MODEL_NAME", "qwen-turbo")
    
    @property
    def embedding_model_name(self) -> str:
        return os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    
    @property
    def langsmith_project(self) -> str:
        return os.environ.get("LANGSMITH_PROJECT", "BidCheck")
    
    @property
    def debug_request_body(self) -> bool:
        """是否开启请求体调试日志"""
        return os.environ.get("DEBUG_REQUEST_BODY", "false").lower() in ("true", "1", "yes", "on")
