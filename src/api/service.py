"""
文档查重核心服务
整合所有功能模块，提供统一的服务接口
"""

import time
import logging
import threading
from typing import List, Dict
from langchain.schema.runnable import RunnableLambda, RunnableParallel

from ..models.api_models import DuplicateOutput
from ..models.data_models import DocumentData
from ..core.document_processor import DocumentProcessor
from ..core.clustering_manager import ClusteringManager
from ..detectors.llm_duplicate_detector import LLMDuplicateDetector
from ..validators.validation_manager import ValidationManager

logger = logging.getLogger(__name__)


class DocumentDeduplicationService:
    """文档查重服务"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.clustering_manager = ClusteringManager()
        self.detector = LLMDuplicateDetector()
        self.validator = ValidationManager()
        self._execution_lock = threading.Lock()
        self._is_running = False
    
    def analyze_documents(self, json_input: List[Dict]) -> List[DuplicateOutput]:
        """分析文档重复内容 - 并行处理分割聚类查重和直接查重"""
        
        # 检查是否已在执行
        if self._is_running:
            logger.warning("工作流正在执行中，请等待完成...")
            return []
        
        with self._execution_lock:
            if self._is_running:
                return []
            
            self._is_running = True
            execution_id = int(time.time() * 1000)
            logger.info(f"开始执行工作流 (ID: {execution_id})")
            
            try:
                # 1. 处理输入数据
                logger.info(f"[{execution_id}] 正在处理JSON输入...")
                document_data_list, document_inputs = self.processor.process_json_documents(json_input)
                
                # 2. 并行执行两种策略
                logger.info(f"[{execution_id}] 🚀 开始并行执行分割聚类查重和直接查重...")
                
                def clustering_strategy():
                    """分割聚类查重策略"""
                    try:
                        # 分割文档
                        segments = self.processor.segment_documents(document_inputs)
                        logger.info(f"[{execution_id}] 聚类策略：已分割出 {len(segments)} 个文本片段")
                        
                        # 生成嵌入向量
                        segments = self.processor.generate_embeddings(segments)
                        logger.info(f"[{execution_id}] 聚类策略：已生成 {len(segments)} 个嵌入向量")
                        
                        # 聚类分析
                        clusters = self.clustering_manager.initial_clustering(segments)
                        multi_doc_clusters = self.clustering_manager.filter_multi_document_clusters(clusters)
                        logger.info(f"[{execution_id}] 聚类策略：发现 {len(multi_doc_clusters)} 个可能包含重复内容的聚类")
                        
                        # 检测重复内容
                        if multi_doc_clusters:
                            cluster_results = self.detector.detect_duplicates_parallel(multi_doc_clusters)
                        else:
                            cluster_results = []
                        
                        logger.info(f"[{execution_id}] 聚类策略：发现 {len(cluster_results)} 对重复内容")
                        return cluster_results
                        
                    except Exception as e:
                        logger.error(f"[{execution_id}] 聚类策略失败: {e}")
                        return []
                
                def direct_strategy():
                    """直接查重策略"""
                    try:
                        direct_results = self.detector.direct_document_comparison(document_data_list)
                        logger.info(f"[{execution_id}] 直接策略：发现 {len(direct_results)} 对重复内容")
                        return direct_results
                        
                    except Exception as e:
                        logger.error(f"[{execution_id}] 直接策略失败: {e}")
                        return []
                
                # 使用RunnableParallel并行执行两种策略
                parallel_tasks = {
                    "clustering": RunnableLambda(lambda x: clustering_strategy()),
                    "direct": RunnableLambda(lambda x: direct_strategy())
                }
                
                parallel_runner = RunnableParallel(parallel_tasks)
                
                try:
                    results = parallel_runner.invoke({})
                    cluster_results = results.get("clustering", [])
                    direct_results = results.get("direct", [])
                except Exception as e:
                    logger.error(f"[{execution_id}] 并行执行失败，回退到串行: {e}")
                    cluster_results = clustering_strategy()
                    direct_results = direct_strategy()
                
                # 3. 合并并去重结果
                logger.info(f"[{execution_id}] 合并结果：聚类 {len(cluster_results)} + 直接 {len(direct_results)}")
                combined_results = cluster_results + direct_results
                unique_results = self._deduplicate_results(combined_results)
                
                # 4. 验证结果
                if unique_results:
                    logger.info(f"[{execution_id}] 🔍 开始验证检测结果...")
                    validated_results = self.validator.validate_results(document_data_list, unique_results)
                    logger.info(f"[{execution_id}] 验证完成，最终结果: {len(validated_results)} 对重复内容")
                else:
                    validated_results = []
                
                return validated_results
                
            except Exception as e:
                logger.error(f"[{execution_id}] 文档分析失败: {e}")
                raise
            finally:
                self._is_running = False
    
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
