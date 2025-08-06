"""
聚类管理器
负责文档片段的聚类和聚类管理
"""

import logging
import numpy as np
import hdbscan
from typing import List, Dict

from ..models.data_models import TextSegment

logger = logging.getLogger(__name__)


class ClusteringManager:
    """聚类管理器"""
    
    def __init__(self, max_cluster_size: int = 10):
        self.max_cluster_size = max_cluster_size
        self.cluster_counter = 0
    
    def initial_clustering(self, segments: List[TextSegment]) -> Dict[int, List[TextSegment]]:
        """初始聚类"""
        if not segments:
            raise ValueError("No segments provided")
        
        # 检查是否有嵌入向量
        segments_with_embeddings = [seg for seg in segments if seg.embedding is not None]
        if not segments_with_embeddings:
            raise ValueError("Segments must have embeddings")
        
        # 提取嵌入向量
        embeddings = np.array([seg.embedding for seg in segments_with_embeddings])
        
        # 使用HDBSCAN进行聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(embeddings)

        # 组织聚类结果
        clusters = {}
        noise_count = 0
        
        # 修改此处：将所有噪声点都归为第0类
        for segment, label in zip(segments_with_embeddings, cluster_labels):
            # 将np.int64转换为int
            label = int(label)
            if label == -1:  # 噪声点统一归为第0类
                label = 0
                noise_count += 1
            else:
                # 非噪声点的聚类ID加1，避免与噪声点的0冲突
                label = label + 1

            segment.cluster_id = label

            if label not in clusters:
                clusters[label] = []
            clusters[label].append(segment)
        
        # 更新cluster_counter
        if clusters:
            self.cluster_counter = max(clusters.keys()) + 1
        
        logger.info(f"初始聚类结果：{len(clusters)} 个聚类，其中噪声点聚类（ID=0）包含 {noise_count} 个点")
        
        return clusters
    
    def filter_multi_document_clusters(self, clusters: Dict[int, List[TextSegment]]) -> Dict[int, List[TextSegment]]:
        """过滤出包含多个文档的聚类，自动跳过噪声点聚类（ID=0）"""
        multi_doc_clusters = {}
        
        for cluster_id, segments in clusters.items():
            if cluster_id == 0:
                continue
            doc_ids = set(seg.document_id for seg in segments)
            if len(doc_ids) > 1:  # 包含多个文档的片段
                multi_doc_clusters[cluster_id] = segments
        
        return multi_doc_clusters
