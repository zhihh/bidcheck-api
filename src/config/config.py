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
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        
        # 模型配置
        os.environ["LLM_MODEL_NAME"] = os.getenv("LLM_MODEL_NAME", "qwen-turbo")
        os.environ["EMBEDDING_MODEL_NAME"] = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
        
        # 聚类配置
        os.environ["CLUSTERING_STRATEGY"] = os.getenv("CLUSTERING_STRATEGY", "enhanced")  # "legacy" 或 "enhanced"
        os.environ["SIMILARITY_THRESHOLD"] = os.getenv("SIMILARITY_THRESHOLD", "0.6")
        os.environ["TOP_K_CANDIDATES"] = os.getenv("TOP_K_CANDIDATES", "8")
        os.environ["USE_RERANKER"] = os.getenv("USE_RERANKER", "true")
        os.environ["MAX_RERANK_CANDIDATES"] = os.getenv("MAX_RERANK_CANDIDATES", "4")
        
        # DashScope 配置（for reranker）
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")
        
        # 验证管理器配置
        os.environ["VALIDATION_STRATEGY"] = os.getenv("VALIDATION_STRATEGY", "optimized")  # "legacy" 或 "optimized"
        os.environ["VALIDATION_SIMILARITY_THRESHOLD"] = os.getenv("VALIDATION_SIMILARITY_THRESHOLD", "0.5")
        os.environ["VALIDATION_VECTOR_THRESHOLD"] = os.getenv("VALIDATION_VECTOR_THRESHOLD", "0.5")
        os.environ["VALIDATION_MAX_WORKERS"] = os.getenv("VALIDATION_MAX_WORKERS", "16")
        os.environ["VALIDATION_USE_GPU"] = os.getenv("VALIDATION_USE_GPU", "true")
        
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
    
    # 新增聚类相关配置属性
    @property
    def clustering_strategy(self) -> str:
        """聚类策略"""
        return os.environ.get("CLUSTERING_STRATEGY", "enhanced")
    
    @property
    def similarity_threshold(self) -> float:
        """相似度阈值"""
        return float(os.environ.get("SIMILARITY_THRESHOLD", "0.7"))
    
    @property
    def top_k_candidates(self) -> int:
        """TOP-K候选数量"""
        return int(os.environ.get("TOP_K_CANDIDATES", "10"))
    
    @property
    def use_reranker(self) -> bool:
        """是否使用reranker"""
        return os.environ.get("USE_RERANKER", "true").lower() in ("true", "1", "yes", "on")
    
    @property
    def max_rerank_candidates(self) -> int:
        """最大rerank候选数"""
        return int(os.environ.get("MAX_RERANK_CANDIDATES", "20"))
    
    @property
    def dashscope_api_key(self) -> str:
        """DashScope API密钥"""
        return os.environ.get("DASHSCOPE_API_KEY", "")
    
    # 验证管理器相关配置属性
    @property
    def validation_strategy(self) -> str:
        """验证策略"""
        return os.environ.get("VALIDATION_STRATEGY", "optimized")
    
    @property
    def validation_similarity_threshold(self) -> float:
        """验证相似度阈值"""
        return float(os.environ.get("VALIDATION_SIMILARITY_THRESHOLD", "0.7"))
    
    @property
    def validation_vector_threshold(self) -> float:
        """验证向量相似度阈值"""
        return float(os.environ.get("VALIDATION_VECTOR_THRESHOLD", "0.95"))
    
    @property
    def validation_max_workers(self) -> int:
        """验证最大工作线程数"""
        return int(os.environ.get("VALIDATION_MAX_WORKERS", "16"))
    
    @property
    def validation_use_gpu(self) -> bool:
        """验证是否使用GPU"""
        return os.environ.get("VALIDATION_USE_GPU", "true").lower() in ("true", "1", "yes", "on")
