"""
文档处理器
负责文档的分割、向量化等预处理工作
"""

import os
import logging
from typing import List, Tuple, Dict
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..models.api_models import DocumentInput
from ..models.data_models import TextSegment, DocumentData

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        # 初始化LangChain文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 每个片段的字符数
            chunk_overlap=50,  # 滑动窗口重叠字符数
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 分割符号优先级
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
        """基于LangChain滑动窗口分割文档，支持跨页面分割"""
        all_segments = []
        
        # 按文档ID分组
        docs_by_id = {}
        for doc_input in document_inputs:
            if doc_input.documentId not in docs_by_id:
                docs_by_id[doc_input.documentId] = []
            docs_by_id[doc_input.documentId].append(doc_input)
        
        # 对每个文档进行分割
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
            
            # 使用LangChain分割器
            langchain_doc = Document(
                page_content=combined_content,
                metadata={"document_id": doc_id}
            )
            
            chunks = self.text_splitter.split_documents([langchain_doc])
            
            # 转换为TextSegment对象，确定每个片段所属的页面
            for chunk_id, chunk in enumerate(chunks, 1):
                chunk_content = chunk.page_content.strip()
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
                
                segment_id = f"doc_{doc_id}_page_{page_num}_chunk_{chunk_id}"
                
                segment = TextSegment(
                    id=segment_id,
                    content=chunk_content,
                    document_id=doc_id,
                    page=page_num,
                    chunk_id=chunk_id
                )
                
                all_segments.append(segment)
        
        logger.info(f"使用LangChain滑动窗口分割，共生成 {len(all_segments)} 个文本片段")
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
