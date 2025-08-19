"""
基于大模型的重复检测器
使用LangChain和RunnableParallel进行并行检测
"""

import json
import os
import time
import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel

from ..models.api_models import DuplicateOutput
from ..models.data_models import TextSegment, DocumentData
from ..utils.text_utils import extract_prefix_suffix

logger = logging.getLogger(__name__)


class LLMDuplicateDetector:
    """基于大模型的重复检测器 - 使用RunnableParallel并行执行"""
    
    def __init__(self):
        # 从环境变量获取模型名
        llm_model_name = os.environ.get("LLM_MODEL_NAME", "qwen-plus")
        
        self.llm = ChatOpenAI(
            model=llm_model_name, # type:ignore
            temperature=0.1
        )
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
        # 创建直接比较的提示模板
        self.direct_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_direct_system_prompt()),
            ("human", self._get_direct_human_prompt())
        ])
        
        # 创建基础链
        self.base_chain = self.prompt | self.llm
        self.direct_chain = self.direct_prompt | self.llm
    
    def _get_system_prompt(self) -> str:
        return """你是一个专业的文档分析系统。你的任务是检测给定文本片段中的三种问题类型，并输出JSON格式的结果。

**检测类型：**
类别1 - 语义相似/重复：
- 完全相同的文本（逐字相同）
- 高度相似的文本（仅有微小差异，如标点、空格、同义词替换）
- 意思相同但表达略有不同的文本
- 结构相似、逻辑相同的文本

类别2 - 错误一致：
- 相同的错别字（如"登录"都写成"登陆"）
- 相同的用词错误（如专业术语使用错误但错误方式一致）
- 相同的语法错误
- 相同的格式错误

类别3 - 报价异常：
- 投标报价呈等差数列规律（如100万、200万、300万...）
- 投标报价呈等比数列规律（如100万、200万、400万...）
- 报价数值过于接近或有明显规律性
- 报价结构异常相似

**评分标准：**
- 0.9-1.0: 问题非常明显
- 0.7-0.9: 问题较明显，需要关注
- 0.5-0.7: 中等程度问题，需要人工确认
- 0.3-0.5: 轻微问题
- 0.0-0.3: 无问题

请仔细分析每对文本片段，识别问题类型并给出准确的判断，找出所有评分达到0.8分以上的问题内容。"""

    def _get_human_prompt(self) -> str:
        return """请分析以下来自不同文档的文本片段，检测三种问题类型：

{text_segments}

请严格按照以下JSON格式返回结果，不要添加任何其他文字：
{{
  "is_duplicate": true/false,
  "duplicate_pairs": [
    {{
      "documentId1": 数字,
      "page1": 数字,
      "chunkId1": 数字,
      "content1": "文本内容",
      "documentId2": 数字,
      "page2": 数字,
      "chunkId2": 数字,
      "content2": "文本内容",
      "reason": "判断原因",
      "score": 数字,
      "category": 数字
    }}
  ],
  "explanation": "整体判断说明"
}}

注意：
1. 只比较来自不同文档的片段
2. category字段：1-语义相似，2-错误一致，3-报价异常
3. 如果发现问题，请在duplicate_pairs中详细列出每一对问题内容
4. 每个duplicate_pair必须包含完整的documentId, page, chunkId, content, category信息
5. explanation应该详细说明判断依据和问题类型
6. 返回的必须是有效的JSON格式，不要包含其他文字"""
    
    def _get_direct_system_prompt(self) -> str:
        """直接比较两个完整文档的系统提示"""
        return """你是一个专业的文档分析系统。你将接收到两个完整的文档，需要找出其中的三种问题类型。

**检测类型：**
类别1 - 语义相似/重复：
- 完全相同的文本段落或句子
- 高度相似的文本（仅有微小差异，如标点、空格、同义词替换）  
- 意思相同但表达略有不同的文本段落
- 结构相似、逻辑相同的段落

类别2 - 错误一致：
- 相同的错别字（如"登录"都写成"登陆"）
- 相同的用词错误
- 相同的语法错误
- 相同的格式错误

类别3 - 报价异常：
- 投标报价呈等差数列或等比数列规律
- 报价数值过于接近或有明显规律性
- 报价结构异常相似

**评分标准：**
- 0.9-1.0: 问题非常明显
- 0.7-0.9: 问题较明显，需要关注
- 0.5-0.7: 中等程度问题，需要人工确认

请仔细分析两个文档，找出所有评分达到0.8分以上的问题内容。"""
    
    def _get_direct_human_prompt(self) -> str:
        """直接比较两个完整文档的人类提示"""
        return """请分析以下两个完整文档，找出其中的三种问题类型：

**文档1:**
{document1}

**文档2:**  
{document2}

请将每个问题的文本片段标记出来，并严格按照以下JSON格式返回结果：
{{
  "is_duplicate": true/false,
  "duplicate_pairs": [
    {{
      "content1": "文档1中的问题片段",
      "content2": "文档2中的问题片段", 
      "reason": "问题说明",
      "score": 数字,
      "category": 数字
    }}
  ],
  "explanation": "整体分析说明"
}}

注意：
1. category字段：1-语义相似，2-错误一致，3-报价异常
2. 对于每个发现的问题，详细说明原因
3. 返回有效的JSON格式，不要添加其他文字"""
    
    def detect_duplicates_parallel(self, clusters_dict: Dict[int, List[TextSegment]]) -> List[DuplicateOutput]:
        """并行检测多个聚类中的重复内容"""
        if not clusters_dict:
            return []
        
        logger.info(f"🚀 开始并行处理 {len(clusters_dict)} 个聚类...")
        start_time = time.time()
        
        # 为每个聚类创建处理函数
        def create_cluster_processor(cluster_id: int, segments: List[TextSegment]):
            def process_cluster(input_data):
                logger.info(f"  🔍 处理聚类 {cluster_id} ({len(segments)} 个片段)")
                return self.detect_duplicates(segments)
            return RunnableLambda(process_cluster)
        
        # 创建并行处理管道
        parallel_tasks = {}
        for cluster_id, segments in clusters_dict.items():
            task_name = f"cluster_{cluster_id}"
            parallel_tasks[task_name] = create_cluster_processor(cluster_id, segments)
        
        # 使用RunnableParallel执行
        parallel_runner = RunnableParallel(parallel_tasks)
        
        try:
            # 执行并行任务
            results = parallel_runner.invoke(
                {}, 
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            # 合并所有结果
            all_duplicate_results = []
            for task_name, task_results in results.items():
                cluster_id = task_name.replace("cluster_", "")
                if task_results:
                    logger.info(f"    ✅ 聚类 {cluster_id} 发现 {len(task_results)} 对重复内容")
                    all_duplicate_results.extend(task_results)
                else:
                    logger.info(f"    ✅ 聚类 {cluster_id} 未发现重复内容")
            
            end_time = time.time()
            logger.info(f"⚡ 并行处理完成，耗时 {end_time - start_time:.2f} 秒")
            logger.info(f"📊 总共发现 {len(all_duplicate_results)} 对重复内容")
            
            return all_duplicate_results
            
        except Exception as e:
            logger.error(f"❌ 并行处理失败: {e}")
            # 回退到串行处理
            return self._fallback_serial_processing(clusters_dict)
    
    def _fallback_serial_processing(self, clusters_dict: Dict[int, List[TextSegment]]) -> List[DuplicateOutput]:
        """回退的串行处理方案"""
        logger.info("🔄 回退到串行处理...")
        all_results = []
        
        for cluster_id, segments in clusters_dict.items():
            logger.info(f"  🔍 串行处理聚类 {cluster_id}")
            results = self.detect_duplicates(segments)
            if results:
                all_results.extend(results)
        
        return all_results
    
    def detect_duplicates(self, segments: List[TextSegment]) -> List[DuplicateOutput]:
        """检测文本片段中的重复内容，返回结构化结果"""
        
        # 格式化输入文本
        text_segments = self._format_segments(segments)
        
        # 调用大模型链
        try:
            response = self.base_chain.invoke({"text_segments": text_segments})
            
            # 解析响应内容
            result_dict = self._parse_response(response.content) # type:ignore
            
            if result_dict and result_dict.get("is_duplicate") and result_dict.get("duplicate_pairs"):
                # 转换为 DuplicateOutput 对象
                duplicate_outputs = []
                for pair in result_dict["duplicate_pairs"]:
                    try:
                        content1 = str(pair["content1"])
                        content2 = str(pair["content2"])
                        prefix1, suffix1 = extract_prefix_suffix(content1)
                        prefix2, suffix2 = extract_prefix_suffix(content2)
                        output = DuplicateOutput(
                            documentId1=int(pair["documentId1"]),
                            page1=int(pair["page1"]),
                            chunkId1=int(pair["chunkId1"]),
                            content1=content1,
                            prefix1=prefix1,
                            suffix1=suffix1,
                            documentId2=int(pair["documentId2"]),
                            page2=int(pair["page2"]),
                            chunkId2=int(pair["chunkId2"]),
                            content2=content2,
                            prefix2=prefix2,
                            suffix2=suffix2,
                            reason=str(pair["reason"]),
                            score=float(pair["score"]),
                            category=int(pair.get("category", 1))  # 默认为语义相似
                        )
                        duplicate_outputs.append(output)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"解析重复对时出错: {e}")
                        continue
                        
                return duplicate_outputs
            else:
                return []
                
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            # 直接返回空结果
            return []
    
    def _parse_response(self, response_content: str) -> Optional[Dict]:
        """解析 LLM 响应"""
        try:
            # 清理响应内容
            content = response_content.strip()
            
            # 尝试提取 JSON
            import re
            
            # 查找 JSON 内容（包括多行）
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # 如果没有找到花括号，尝试整个内容
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应内容: {response_content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"响应解析出错: {e}")
            return None
    
    def direct_document_comparison(self, document_data_list: List[DocumentData]) -> List[DuplicateOutput]:
        """直接比较策略：直接比较完整文档内容"""
        logger.info("🔄 启动直接比较策略：直接比较完整文档")
        
        if len(document_data_list) < 2:
            return []
        
        results = []
        
        # 两两比较所有文档
        for i, doc1 in enumerate(document_data_list):
            for j, doc2 in enumerate(document_data_list[i+1:], i+1):
                if doc1.document_id != doc2.document_id:  # 确保比较不同文档
                    try:
                        # 调用直接比较链
                        response = self.direct_chain.invoke({
                            "document1": doc1.content,
                            "document2": doc2.content
                        }, 
                            model_kwargs={
                            "response_format": {"type": "json_object"}
                            }
                        )

                        # 解析响应
                        result_dict = self._parse_response(response.content) # type:ignore
                        if result_dict and result_dict.get("is_duplicate") and result_dict.get("duplicate_pairs"):
                            for pair in result_dict["duplicate_pairs"]:
                                try:
                                    # 获取内容在各自文档中的精确页面信息
                                    content1 = str(pair["content1"])
                                    content2 = str(pair["content2"])
                                    
                                    page1 = self._find_content_page(content1, doc1)
                                    page2 = self._find_content_page(content2, doc2)
                                    
                                    prefix1, suffix1 = extract_prefix_suffix(content1)
                                    prefix2, suffix2 = extract_prefix_suffix(content2)
                                    
                                    # 创建输出结果，使用精确的页面信息和固定的chunk_id
                                    output = DuplicateOutput(
                                        documentId1=doc1.document_id,
                                        page1=page1,  # 精确的页面信息
                                        chunkId1=0,   # 固定的chunk_id，表示直接比较
                                        content1=content1,
                                        prefix1=prefix1,
                                        suffix1=suffix1,
                                        documentId2=doc2.document_id,
                                        page2=page2,  # 精确的页面信息
                                        chunkId2=0,   # 固定的chunk_id，表示直接比较
                                        content2=content2,
                                        prefix2=prefix2,
                                        suffix2=suffix2,
                                        reason=str(pair['reason']),
                                        score=float(pair["score"]),
                                        category=int(pair.get("category", 1))  # 默认为语义相似
                                    )

                                    results.append(output)
                                except (KeyError, ValueError, TypeError) as e:
                                    logger.warning(f"解析直接比较结果时出错: {e}")
                                    continue
                    
                    except Exception as e:
                        logger.error(f"直接文档比较失败: {e}")
                        continue
        
        logger.info(f"🔍 直接比较发现 {len(results)} 对重复内容")
        return results
    
    def _find_content_page(self, content: str, document_data: DocumentData) -> int:
        """在文档的页面中找到内容所在的页面"""
        content = content.strip()
        
        # 遍历所有页面，找到包含该内容的页面
        for page_num, page_content in document_data.pages.items():
            if content in page_content:
                return page_num
        
        # 如果在单个页面中找不到，可能是跨页面的内容，返回第一个包含部分内容的页面
        for page_num, page_content in document_data.pages.items():
            # 检查内容的前半部分或后半部分是否在某个页面中
            content_length = len(content)
            content_start = content[:content_length // 2]
            content_end = content[content_length // 2:]
            
            if content_start in page_content or content_end in page_content:
                return page_num
        
        # 如果都找不到，返回第一个页面
        return min(document_data.pages.keys()) if document_data.pages else 1
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
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
    
    def _format_segments(self, segments: List[TextSegment]) -> str:
        """格式化文本片段用于输入"""
        formatted = []
        
        for i, segment in enumerate(segments):
            formatted.append(
                f"**片段 {i+1}** (文档ID: {segment.document_id}, 页码: {segment.page}, 片段ID: {segment.chunk_id}):\n"
                f"{segment.content}\n"
            )
        
        return "\n".join(formatted)
