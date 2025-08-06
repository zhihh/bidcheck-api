"""
验证管理器
基于规则的溯源验证逻辑
"""

import logging
from typing import List, Dict, Any

from ..models.api_models import DuplicateOutput
from ..models.data_models import DocumentData

logger = logging.getLogger(__name__)


class ValidationManager:
    """验证管理器 - 基于规则的溯源验证"""
    
    def __init__(self):
        pass
    
    def validate_results(self, document_data_list: List[DocumentData], 
                        detected_results: List[DuplicateOutput]) -> List[DuplicateOutput]:
        """基于规则的验证检测结果"""
        if not detected_results:
            return detected_results
        
        logger.info("🔍 开始验证检测结果...")
        
        # 基于规则的验证
        rule_validated = self._rule_based_validation(document_data_list, detected_results)
        
        logger.info(f"✅ 验证完成，保留 {len(rule_validated)} 对重复内容")
        return rule_validated
    
    def _rule_based_validation(self, document_data_list: List[DocumentData], 
                              results: List[DuplicateOutput]) -> List[DuplicateOutput]:
        """基于规则的验证 - 检查content字段是否来自原文而不是大模型的幻觉"""
        logger.info("📏 开始基于规则的溯源验证...")
        
        # 创建文档内容查找字典
        doc_dict = {doc.document_id: doc.content for doc in document_data_list}
        
        validated_results = []
        
        for result in results:
            # 获取原文档内容
            doc1_content = doc_dict.get(result.documentId1, "")
            doc2_content = doc_dict.get(result.documentId2, "")
            
            if not doc1_content or not doc2_content:
                logger.warning(f"验证失败：找不到文档内容 (doc1: {result.documentId1}, doc2: {result.documentId2})")
                continue
            
            # 溯源验证：检查content字段在原文中的位置并计算匹配度
            content1_match = self._find_best_match_in_source(result.content1, doc1_content)
            content2_match = self._find_best_match_in_source(result.content2, doc2_content)
            
            # 设定相似度阈值
            similarity_threshold = 0.7
            
            if content1_match['similarity'] < similarity_threshold or content2_match['similarity'] < similarity_threshold:
                logger.warning(f"溯源验证失败：content与原文匹配度不足 (content1: {content1_match['similarity']:.3f}, content2: {content2_match['similarity']:.3f})")
                continue
            
            # 验证通过：用原文中对应的内容替换检测结果中的content
            validated_result = DuplicateOutput(
                documentId1=result.documentId1,
                page1=result.page1,
                chunkId1=result.chunkId1,
                content1=content1_match['matched_text'],  # 使用原文中的内容
                documentId2=result.documentId2,
                page2=result.page2,
                chunkId2=result.chunkId2,
                content2=content2_match['matched_text'],  # 使用原文中的内容
                reason=result.reason,
                score=result.score
            )
            
            # 额外验证：检查内容长度合理性
            if len(validated_result.content1.strip()) < 10 or len(validated_result.content2.strip()) < 10:
                logger.warning(f"验证失败：内容过短")
                continue
            
            # 额外验证：检查相似度分数合理性
            if result.score < 0.3 or result.score > 1.0:
                logger.warning(f"验证失败：相似度分数不合理 ({result.score})")
                continue
            
            validated_results.append(validated_result)
        
        logger.info(f"📏 规则验证完成，保留 {len(validated_results)}/{len(results)} 对结果")
        return validated_results
    
    def _find_best_match_in_source(self, content: str, source_document: str) -> Dict[str, Any]:
        """在源文档中找到与content最匹配的片段"""
        content = content.strip()
        
        # 方法1：精确匹配
        if content in source_document:
            return {
                'matched_text': content,
                'similarity': 1.0,
                'method': 'exact_match'
            }
        
        # 方法2：子字符串匹配 
        best_match = {
            'matched_text': content,
            'similarity': 0.0,
            'method': 'substring_match'
        }
        
        # 将文档分割成句子进行匹配
        import re
        sentences = re.split(r'[。！？；\n]+', source_document)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        # 查找最佳匹配的句子
        for sentence in sentences:
            similarity = self._calculate_string_similarity(content, sentence)
            if similarity > best_match['similarity']:
                best_match = {
                    'matched_text': sentence,
                    'similarity': similarity,
                    'method': 'sentence_match'
                }
        
        # 方法3：滑动窗口匹配
        window_size = len(content)
        for i in range(0, len(source_document) - window_size + 1, 10):  # 步长为10
            window_text = source_document[i:i + window_size]
            similarity = self._calculate_string_similarity(content, window_text)
            if similarity > best_match['similarity']:
                best_match = {
                    'matched_text': window_text,
                    'similarity': similarity,
                    'method': 'window_match'
                }
        
        return best_match
    
    def _calculate_string_similarity(self, text1: str, text2: str) -> float:
        """计算两个字符串的相似度"""
        if not text1 or not text2:
            return 0.0
        
        if text1 == text2:
            return 1.0
        
        # 使用编辑距离计算相似度
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(text1), len(text2))
        distance = levenshtein_distance(text1, text2)
        similarity = 1 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """计算简单的文本相似度"""
        if text1 == text2:
            return 1.0
        
        # 基于字符的相似度
        longer = text1 if len(text1) > len(text2) else text2
        shorter = text2 if len(text1) > len(text2) else text1
        
        if len(longer) == 0:
            return 1.0
        
        # 计算编辑距离的简化版本
        matches = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matches / len(longer)
