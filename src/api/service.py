"""
文档查重核心服务 - 高并发版本
整合所有功能模块，提供统一的服务接口，支持并发处理
"""

import time
import logging
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema.runnable import RunnableLambda, RunnableParallel

from ..models.api_models import DuplicateOutput
from ..models.data_models import DocumentData
from ..core.document_processor import DocumentProcessor
from ..core.clustering_manager import ClusteringManager
from ..detectors.llm_duplicate_detector import LLMDuplicateDetector
from ..validators.validation_manager import ValidationManager

logger = logging.getLogger(__name__)


class DocumentDeduplicationService:
    """文档查重服务 - 高并发版本"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.clustering_manager = ClusteringManager()
        self.detector = LLMDuplicateDetector()
        self.validator = ValidationManager()
        # 移除全局锁，支持并发处理
        self.max_workers = 4  # 可根据服务器配置调整
    
    async def analyze_documents(self, json_input: List[Dict]) -> List[DuplicateOutput]:
        """分析文档重复内容 - 高并发异步处理版本"""
        
        execution_id = int(time.time() * 1000)
        logger.info(f"开始执行工作流 (ID: {execution_id})")
        
        try:
            # 1. 处理输入数据
            logger.info(f"[{execution_id}] 正在处理JSON输入...")
            document_data_list, document_inputs = await self._run_in_executor(
                self.processor.process_json_documents, json_input
            )
            
            # 2. 并行执行两种策略
            logger.info(f"[{execution_id}] 🚀 开始并行执行分割聚类查重和直接查重...")
            
            # 使用asyncio创建并发任务
            cluster_task = asyncio.create_task(
                self._clustering_strategy(execution_id, document_inputs)
            )
            direct_task = asyncio.create_task(
                self._direct_strategy(execution_id, document_data_list)
            )
            
            # 等待两个任务完成
            cluster_results, direct_results = await asyncio.gather(
                cluster_task, 
                direct_task, 
                return_exceptions=True
            )
            
            # 处理异常结果
            if isinstance(cluster_results, Exception):
                logger.error(f"[{execution_id}] 聚类策略失败: {cluster_results}")
                cluster_results = []
            
            if isinstance(direct_results, Exception):
                logger.error(f"[{execution_id}] 直接策略失败: {direct_results}")
                direct_results = []
            
            # 确保结果是列表类型
            cluster_results = cluster_results if isinstance(cluster_results, list) else []
            direct_results = direct_results if isinstance(direct_results, list) else []
            
            # 3. 合并并去重结果
            logger.info(f"[{execution_id}] 合并结果：聚类 {len(cluster_results)} + 直接 {len(direct_results)}")
            combined_results = cluster_results + direct_results
            unique_results = self._deduplicate_results(combined_results)
            
            # 4. 验证结果
            if unique_results:
                logger.info(f"[{execution_id}] 🔍 开始验证检测结果...")
                validated_results = await self._run_in_executor(
                    self.validator.validate_results, document_data_list, unique_results
                )
                logger.info(f"[{execution_id}] 验证完成，最终结果: {len(validated_results)} 对重复内容")
            else:
                validated_results = []
            
            return validated_results
            
        except Exception as e:
            logger.error(f"[{execution_id}] 文档分析失败: {e}")
            raise
    
    async def _clustering_strategy(self, execution_id: int, document_inputs) -> List[DuplicateOutput]:
        """分割聚类查重策略 - 异步版本"""
        try:
            # 分割文档
            segments = await self._run_in_executor(
                self.processor.segment_documents, document_inputs
            )
            logger.info(f"[{execution_id}] 聚类策略：已分割出 {len(segments)} 个文本片段")
            
            # 生成嵌入向量
            segments = await self._run_in_executor(
                self.processor.generate_embeddings, segments
            )
            logger.info(f"[{execution_id}] 聚类策略：已生成 {len(segments)} 个嵌入向量")
            
            # 聚类分析
            clusters = await self._run_in_executor(
                self.clustering_manager.initial_clustering, segments
            )
            multi_doc_clusters = await self._run_in_executor(
                self.clustering_manager.filter_multi_document_clusters, clusters
            )
            logger.info(f"[{execution_id}] 聚类策略：发现 {len(multi_doc_clusters)} 个可能包含重复内容的聚类")
            
            # 检测重复内容
            if multi_doc_clusters:
                cluster_results = await self._run_in_executor(
                    self.detector.detect_duplicates_parallel, multi_doc_clusters
                )
            else:
                cluster_results = []
            
            logger.info(f"[{execution_id}] 聚类策略：发现 {len(cluster_results)} 对重复内容")
            return cluster_results
            
        except Exception as e:
            logger.error(f"[{execution_id}] 聚类策略失败: {e}")
            return []
    
    async def _direct_strategy(self, execution_id: int, document_data_list) -> List[DuplicateOutput]:
        """直接查重策略 - 异步版本"""
        try:
            direct_results = await self._run_in_executor(
                self.detector.direct_document_comparison, document_data_list
            )
            logger.info(f"[{execution_id}] 直接策略：发现 {len(direct_results)} 对重复内容")
            return direct_results
            
        except Exception as e:
            logger.error(f"[{execution_id}] 直接策略失败: {e}")
            return []
    
    async def _run_in_executor(self, func, *args):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return await loop.run_in_executor(executor, func, *args)
    
    def _deduplicate_results(self, results: List[DuplicateOutput]) -> List[DuplicateOutput]:
        """去除重复的检测结果"""
        if not results:
            return results
        
        unique_results = []
        seen_pairs = set()
        
        for result in results:
            # 创建标准化的内容对标识
            content_pair = tuple(sorted([
                result.content1.strip().lower(),
                result.content2.strip().lower()
            ]))
            
            if content_pair not in seen_pairs:
                seen_pairs.add(content_pair)
                unique_results.append(result)
        
        logger.info(f"去重前: {len(results)} 对，去重后: {len(unique_results)} 对")
        return unique_results
