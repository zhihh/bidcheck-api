"""
文本处理工具
"""

import re
from typing import Tuple, Optional


def extract_prefix_suffix(content: str, n: int = 10) -> Tuple[str, str]:
    """
    从内容中提取前缀和后缀
    
    Args:
        content: 原始内容
        n: 提取的字符数，默认10字
        
    Returns:
        (前缀, 后缀) 元组
    """
    if not content or not content.strip():
        return "", ""
    
    # 清理内容，去除首尾空白
    cleaned_content = content.strip()
    content_length = len(cleaned_content)
    
    # 如果内容长度小于等于2*n，则前缀为前一半，后缀为后一半
    if content_length <= 2 * n:
        mid_point = content_length // 2
        prefix = cleaned_content[:mid_point]
        suffix = cleaned_content[mid_point:]
        return prefix, suffix
    
    # 提取前缀（前n个字符）
    prefix = cleaned_content[:n]
    
    # 提取后缀（后n个字符）  
    suffix = cleaned_content[-n:]
    
    return prefix, suffix


def generate_content_preview(content: str, max_chars: int = 30) -> str:
    """
    生成内容预览
    
    优先提取第一句话，如果第一句话超过最大字符数，则截取前N个字符
    
    Args:
        content: 原始内容
        max_chars: 最大字符数，默认30
        
    Returns:
        内容预览字符串
    """
    if not content or not content.strip():
        return ""
    
    # 清理内容
    cleaned_content = content.strip()
    
    # 尝试提取第一句话（以句号、问号、感叹号结尾）
    sentence_pattern = r'^([^。？！\n]*[。？！])'
    sentence_match = re.match(sentence_pattern, cleaned_content)
    
    if sentence_match:
        first_sentence = sentence_match.group(1)
        # 如果第一句话长度合适，返回第一句话
        if len(first_sentence) <= max_chars:
            return first_sentence
    
    # 如果没有找到第一句话或第一句话太长，截取前N个字符
    if len(cleaned_content) <= max_chars:
        return cleaned_content
    else:
        # 截取前N个字符，并添加省略号
        return cleaned_content[:max_chars] + "..."


def extract_first_sentence_or_chars(content: str, max_chars: int = 30) -> str:
    """
    提取第一句话或前N个字符
    这是generate_content_preview的别名，保持向后兼容
    """
    return generate_content_preview(content, max_chars)
