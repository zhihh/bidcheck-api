"""
增强版聚类管理器
使用ANN近似召回 + 全量相似度矩阵fallback + Qwen reranker
专门优化两两查重任务的性能
"""

import logging
import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from ..models.data_models import TextSegment

logger = logging.getLogger(__name__)

# 动态导入dashscope以避免依赖问题
try:
    import dashscope
    from http import HTTPStatus
    DASHSCOPE_AVAILABLE = True
except ImportError:
    logger.warning("dashscope未安装，将禁用reranker功能")
    DASHSCOPE_AVAILABLE = False


class ClusteringManager:
    """增强版聚类管理器 - 专为两两查重优化"""
    
    def __init__(self, 
                 top_k: int = 10, 
                 similarity_threshold: float = 0.7,
                 use_reranker: bool = True,
                 max_candidates_for_rerank: int = 20):
        """
        初始化管理器
        
        Args:
            top_k: ANN召回的候选数量
            similarity_threshold: 相似度阈值
            use_reranker: 是否使用reranker精排
            max_candidates_for_rerank: 送入reranker的最大候选数
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_reranker = use_reranker and DASHSCOPE_AVAILABLE
        self.max_candidates_for_rerank = max_candidates_for_rerank
        
        # 初始化reranker客户端
        if self.use_reranker:
            self._init_reranker()
    
    def _init_reranker(self):
        """初始化Qwen reranker"""
        if not DASHSCOPE_AVAILABLE:
            self.use_reranker = False
            return
            
        try:
            # 从环境变量获取API密钥
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("未找到DASHSCOPE_API_KEY，将禁用reranker功能")
                self.use_reranker = False
                return
            
            dashscope.api_key = api_key
            logger.info("已初始化Qwen reranker")
        except Exception as e:
            logger.error(f"初始化reranker失败: {e}")
            self.use_reranker = False
    
    def ann_similarity_search(self, segments: List[TextSegment]) -> Dict[int, List[TextSegment]]:
        """
        基于ANN的相似性搜索
        为每个文档片段找到最相似的候选片段
        """
        if not segments:
            raise ValueError("文档片段列表为空")
        
        # 检查嵌入向量
        segments_with_embeddings = [seg for seg in segments if seg.embedding is not None]
        if not segments_with_embeddings:
            raise ValueError("文档片段必须包含嵌入向量")
        
        logger.info(f"开始ANN相似性搜索，处理 {len(segments_with_embeddings)} 个片段")
        start_time = time.time()
        
        # 构建嵌入矩阵
        embeddings = np.array([seg.embedding for seg in segments_with_embeddings])
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 为每个片段找到相似的候选片段
        candidate_pairs = {}
        
        for i, segment in enumerate(segments_with_embeddings):
            # 获取与当前片段最相似的top_k个片段
            similarities = similarity_matrix[i]
            
            # 排除自己，并按相似度排序
            candidate_indices = []
            for j, sim in enumerate(similarities):
                if i != j and sim >= self.similarity_threshold:
                    # 只考虑来自不同文档的片段
                    if segments_with_embeddings[j].document_id != segment.document_id:
                        candidate_indices.append((j, sim))
            
            # 按相似度排序并取top_k
            candidate_indices.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidate_indices[:self.top_k]
            
            if top_candidates:
                candidates = [segments_with_embeddings[idx] for idx, _ in top_candidates]
                # 使用元组作为键以避免重复
                pair_key = (min(segment.document_id, candidates[0].document_id),
                          max(segment.document_id, candidates[0].document_id))
                
                if pair_key not in candidate_pairs:
                    candidate_pairs[pair_key] = []
                
                candidate_pairs[pair_key].extend([segment] + candidates)
        
        # 转换为聚类格式（为了兼容现有接口）
        clusters = {}
        cluster_id = 1
        
        for pair_key, pair_segments in candidate_pairs.items():
            # 去重
            unique_segments = []
            seen_ids = set()
            for seg in pair_segments:
                seg_id = (seg.document_id, seg.page, seg.chunk_id)
                if seg_id not in seen_ids:
                    seen_ids.add(seg_id)
                    unique_segments.append(seg)
            
            if len(unique_segments) >= 2:
                clusters[cluster_id] = unique_segments
                cluster_id += 1
        
        elapsed = time.time() - start_time
        logger.info(f"ANN搜索完成，耗时 {elapsed:.2f}秒，发现 {len(clusters)} 个候选聚类")
        
        return clusters
    
    def full_similarity_matrix_fallback(self, segments: List[TextSegment]) -> Dict[int, List[TextSegment]]:
        """
        全量相似度矩阵计算作为fallback
        """
        if not segments:
            return {}
        
        logger.info("使用全量相似度矩阵fallback策略")
        start_time = time.time()
        
        segments_with_embeddings = [seg for seg in segments if seg.embedding is not None]
        if not segments_with_embeddings:
            return {}
        
        # 计算全量相似度矩阵
        embeddings = np.array([seg.embedding for seg in segments_with_embeddings])
        similarity_matrix = cosine_similarity(embeddings)
        
        # 找到所有超过阈值的相似对
        clusters = {}
        cluster_id = 1
        processed_pairs = set()
        
        for i in range(len(segments_with_embeddings)):
            for j in range(i + 1, len(segments_with_embeddings)):
                seg1 = segments_with_embeddings[i]
                seg2 = segments_with_embeddings[j]
                
                # 只比较不同文档的片段
                if seg1.document_id != seg2.document_id:
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= self.similarity_threshold:
                        pair_key = tuple(sorted([
                            (seg1.document_id, seg1.page, seg1.chunk_id),
                            (seg2.document_id, seg2.page, seg2.chunk_id)
                        ]))
                        
                        if pair_key not in processed_pairs:
                            processed_pairs.add(pair_key)
                            clusters[cluster_id] = [seg1, seg2]
                            cluster_id += 1
        
        elapsed = time.time() - start_time
        logger.info(f"全量相似度计算完成，耗时 {elapsed:.2f}秒，发现 {len(clusters)} 个相似对")
        
        return clusters
    
    def rerank_candidates(self, query_segment: TextSegment, candidate_segments: List[TextSegment]) -> List[Tuple[TextSegment, float]]:
        """
        使用Qwen reranker对候选片段进行精确排序
        """
        if not self.use_reranker or not candidate_segments or not DASHSCOPE_AVAILABLE:
            # 如果不使用reranker，返回基于余弦相似度的排序
            return [(seg, 0.5) for seg in candidate_segments]
        
        try:
            # 准备文档列表
            documents = [seg.content for seg in candidate_segments]
            
            # 限制候选数量以避免API限制
            if len(documents) > self.max_candidates_for_rerank:
                documents = documents[:self.max_candidates_for_rerank]
                candidate_segments = candidate_segments[:self.max_candidates_for_rerank]
            
            # 调用reranker API
            resp = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query_segment.content,
                documents=documents,
                top_n=len(documents),
                return_documents=True
            )
            
            if resp.status_code == HTTPStatus.OK:
                results = resp.output.results
                
                # 构建排序结果
                reranked_results = []
                for result in results:
                    index = result.index
                    score = result.relevance_score
                    segment = candidate_segments[index]
                    reranked_results.append((segment, score))
                
                logger.debug(f"Reranker成功处理 {len(reranked_results)} 个候选片段")
                return reranked_results
            else:
                logger.warning(f"Reranker调用失败: {resp.message}")
                return [(seg, 0.5) for seg in candidate_segments]
                
        except Exception as e:
            logger.error(f"Reranker调用异常: {e}")
            return [(seg, 0.5) for seg in candidate_segments]
    
    def enhanced_similarity_search(self, segments: List[TextSegment]) -> Dict[int, List[TextSegment]]:
        """
        增强版相似性搜索：ANN + Reranker + Fallback
        """
        if not segments:
            raise ValueError("文档片段列表为空")
        
        logger.info(f"开始增强版相似性搜索，处理 {len(segments)} 个片段")
        
        try:
            # 优先使用ANN搜索
            clusters = self.ann_similarity_search(segments)
            
            # 如果ANN搜索结果太少，使用fallback
            if len(clusters) == 0:
                logger.info("ANN搜索无结果，启用全量相似度矩阵fallback")
                clusters = self.full_similarity_matrix_fallback(segments)
            
            # 使用reranker优化结果
            if self.use_reranker and clusters:
                clusters = self._apply_reranker_to_clusters(clusters)
            
            return clusters
            
        except Exception as e:
            logger.error(f"增强版搜索失败，使用fallback: {e}")
            return self.full_similarity_matrix_fallback(segments)
    
    def _apply_reranker_to_clusters(self, clusters: Dict[int, List[TextSegment]]) -> Dict[int, List[TextSegment]]:
        """
        对聚类结果应用reranker优化
        """
        logger.info("对聚类结果应用reranker优化")
        optimized_clusters = {}
        
        for cluster_id, segments in clusters.items():
            if len(segments) < 2:
                continue
            
            # 将第一个片段作为query，其余作为候选
            query_segment = segments[0]
            candidate_segments = segments[1:]
            
            # 使用reranker排序
            reranked_results = self.rerank_candidates(query_segment, candidate_segments)
            
            # 过滤低分结果（可以根据需要调整阈值）
            rerank_threshold = 0.3
            filtered_results = [
                (seg, score) for seg, score in reranked_results 
                if score >= rerank_threshold
            ]
            
            if filtered_results:
                # 重构聚类，包含query和高分候选
                optimized_segments = [query_segment] + [seg for seg, _ in filtered_results]
                optimized_clusters[cluster_id] = optimized_segments
        
        logger.info(f"Reranker优化完成，优化后聚类数量: {len(optimized_clusters)}")
        return optimized_clusters
    
    def filter_multi_document_clusters(self, clusters: Dict[int, List[TextSegment]]) -> Dict[int, List[TextSegment]]:
        """
        过滤出包含多个文档的聚类（保持与原接口兼容）
        """
        multi_doc_clusters = {}
        
        for cluster_id, segments in clusters.items():
            doc_ids = set(seg.document_id for seg in segments)
            if len(doc_ids) > 1:  # 包含多个文档的片段
                multi_doc_clusters[cluster_id] = segments
        
        return multi_doc_clusters
    
    # 为了保持兼容性，提供原有接口
    def initial_clustering(self, segments: List[TextSegment]) -> Dict[int, List[TextSegment]]:
        """
        初始聚类（兼容接口）
        现在使用增强版相似性搜索
        """
        return self.enhanced_similarity_search(segments)