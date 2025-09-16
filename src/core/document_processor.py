"""
文档处理器
负责文档的分割、向量化等预处理工作
"""

import os
import logging
from typing import List, Tuple, Dict
from openai import OpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from ..models.api_models import DocumentInput
from ..models.data_models import TextSegment, DocumentData

logger = logging.getLogger(__name__)


class CustomEmbeddings(Embeddings):
    """自定义嵌入类，兼容阿里云DashScope API"""
    
    def __init__(self, client: OpenAI, model: str, dimensions: int = 1024):
        self.client = client
        self.model = model
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        all_embeddings = []
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"嵌入生成失败: {e}")
                # 返回零向量作为回退
                for _ in batch_texts:
                    all_embeddings.append([0.0] * self.dimensions)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"查询嵌入生成失败: {e}")
            # 返回零向量作为回退
            return [0.0] * self.dimensions


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        # 使用自定义嵌入类，兼容阿里云DashScope API
        embeddings = CustomEmbeddings(
            client=self.client,
            model=os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v4"),
            dimensions=1024
        )
        self.text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # 使用百分位数阈值
            breakpoint_threshold_amount=95,  # 95%百分位数作为阈值
            buffer_size=1,  # 缓冲区大小
            sentence_split_regex=r'(?<=[。！？；])\s*',  # 中文句子分割正则
        )
        # 从环境变量获取模型名
        self.embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    
    def process_json_documents(self, json_data: List[Dict]) -> Tuple[List[DocumentData], List[DocumentInput]]:
        """处理JSON格式的文档数据，返回按doc为单位的数据和原始输入"""
        documents_by_id = {}
        original_inputs = []
        
        # 按文档ID分组，合并同一文档的所有页面
        for item in json_data:
            doc_input = DocumentInput(
                documentId=item["documentId"],
                page=item["page"],
                content=item["content"]
            )
            original_inputs.append(doc_input)
            
            doc_id = doc_input.documentId
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    'pages': {},
                    'content_parts': []
                }
            
            documents_by_id[doc_id]['pages'][doc_input.page] = doc_input.content
            documents_by_id[doc_id]['content_parts'].append(doc_input.content)
        
        # 创建DocumentData对象
        document_data_list = []
        for doc_id, data in documents_by_id.items():
            combined_content = "\n\n".join(data['content_parts'])
            doc_data = DocumentData(
                document_id=doc_id,
                content=combined_content,
                pages=data['pages']
            )
            document_data_list.append(doc_data)
        
        return document_data_list, original_inputs
    
    def segment_documents(self, document_inputs: List[DocumentInput]) -> List[TextSegment]:
        """基于语义相似性分割文档，支持跨页面的智能分块"""
        all_segments = []
        
        # 按文档ID分组
        docs_by_id = {}
        for doc_input in document_inputs:
            if doc_input.documentId not in docs_by_id:
                docs_by_id[doc_input.documentId] = []
            docs_by_id[doc_input.documentId].append(doc_input)
        
        # 对每个文档进行语义分割
        for doc_id, doc_pages in docs_by_id.items():
            # 按页码排序
            doc_pages.sort(key=lambda x: x.page)
            
            # 合并所有页面内容，记录页面边界
            combined_content = ""
            page_boundaries = []  # [(start_pos, end_pos, page_num), ...]
            current_pos = 0
            
            for doc_page in doc_pages:
                if doc_page.content:
                    start_pos = current_pos
                    end_pos = current_pos + len(doc_page.content)
                    page_boundaries.append((start_pos, end_pos, doc_page.page))
                    combined_content += doc_page.content + "\n\n"
                    current_pos = len(combined_content)
            
            if not combined_content.strip():
                logger.warning(f"文档 {doc_id} 内容为空")
                continue
            
            # 使用语义分块器进行分割
            try:
                chunks = self.text_splitter.split_text(combined_content)
                
                # 转换为TextSegment对象，确定每个片段所属的页面
                for chunk_id, chunk_content in enumerate(chunks, 1):
                    chunk_content = chunk_content.strip()
                    if not chunk_content:
                        continue
                    
                    # 找到片段在合并内容中的位置
                    chunk_start = combined_content.find(chunk_content)
                    if chunk_start == -1:
                        # 如果找不到精确匹配，使用第一个页面
                        page_num = doc_pages[0].page
                    else:
                        # 根据位置确定所属页面（使用片段开始位置所在的页面）
                        page_num = doc_pages[0].page  # 默认值
                        for start_pos, end_pos, page in page_boundaries:
                            if start_pos <= chunk_start < end_pos:
                                page_num = page
                                break
                    
                    segment_id = f"doc_{doc_id}_page_{page_num}_semantic_chunk_{chunk_id}"
                    
                    segment = TextSegment(
                        id=segment_id,
                        content=chunk_content,
                        document_id=doc_id,
                        page=page_num,
                        chunk_id=chunk_id
                    )
                    
                    all_segments.append(segment)
                    
            except Exception as e:
                logger.error(f"对文档 {doc_id} 进行语义分割时出错: {e}")
                # 回退到简单的按句子分割
                sentences = combined_content.split('。')
                for chunk_id, sentence in enumerate(sentences, 1):
                    sentence = sentence.strip()
                    if sentence:
                        segment_id = f"doc_{doc_id}_fallback_chunk_{chunk_id}"
                        segment = TextSegment(
                            id=segment_id,
                            content=sentence + '。',
                            document_id=doc_id,
                            page=doc_pages[0].page,
                            chunk_id=chunk_id
                        )
                        all_segments.append(segment)
        
        logger.info(f"使用语义分块器，共生成 {len(all_segments)} 个文本片段")
        return all_segments
    
    def generate_embeddings(self, segments: List[TextSegment]) -> List[TextSegment]:
        """生成文本嵌入"""
        valid_segments = []
        contents = []
        
        # 也许没有必要
        for seg in segments:
            if seg.content and isinstance(seg.content, str) and seg.content.strip():
                valid_segments.append(seg)
                contents.append(str(seg.content).strip())
        
        if not contents:
            logger.error("没有有效的文本内容用于生成嵌入")
            return segments
        
        logger.info(f"准备为 {len(contents)} 个文本片段生成嵌入...")
        
        try:
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")
                
                response = self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch_contents,
                    dimensions=1024,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            for segment, embedding in zip(valid_segments, all_embeddings):
                segment.embedding = embedding
                
            logger.info(f"成功生成 {len(all_embeddings)} 个嵌入向量")
            
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {e}")
            raise e
        
        return segments
