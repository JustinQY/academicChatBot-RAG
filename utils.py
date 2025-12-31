"""
工具函数模块
提供文件验证、命名策略、哈希计算等工具函数
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, Tuple
import streamlit as st


def generate_unique_filename(original_filename: str) -> str:
    """
    生成唯一的文件名，避免冲突
    
    策略：timestamp_hash_original_name.ext
    
    Args:
        original_filename: 原始文件名
        
    Returns:
        唯一的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
    
    # 分离文件名和扩展名
    name, ext = os.path.splitext(original_filename)
    
    # 清理文件名中的特殊字符
    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))[:50]
    
    return f"{timestamp}_{name_hash}_{safe_name}{ext}"


def calculate_file_hash(file_content: bytes) -> str:
    """
    计算文件内容的 SHA256 哈希值，用于去重
    
    Args:
        file_content: 文件内容（字节）
        
    Returns:
        SHA256 哈希值（十六进制字符串）
    """
    return hashlib.sha256(file_content).hexdigest()


def validate_pdf_file(uploaded_file, max_size_mb: int = 50) -> Tuple[bool, Optional[str]]:
    """
    验证上传的 PDF 文件
    
    Args:
        uploaded_file: Streamlit UploadedFile 对象
        max_size_mb: 最大文件大小（MB）
        
    Returns:
        (是否有效, 错误信息)
    """
    # 检查文件类型
    if uploaded_file.type != "application/pdf":
        return False, "❌ 文件格式错误！请上传 PDF 文件。"
    
    # 检查文件大小
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"❌ 文件过大！文件大小为 {file_size_mb:.2f}MB，最大允许 {max_size_mb}MB。"
    
    # 检查文件名
    if not uploaded_file.name:
        return False, "❌ 文件名无效！"
    
    # 基本内容检查（PDF 魔术数字）
    try:
        file_content = uploaded_file.getvalue()
        if not file_content.startswith(b'%PDF'):
            return False, "❌ 文件内容无效！这不是一个有效的 PDF 文件。"
    except Exception as e:
        return False, f"❌ 无法读取文件内容：{str(e)}"
    
    return True, None


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读格式
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的字符串（例如："1.23 MB"）
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_directory_size(directory: str) -> int:
    """
    计算目录的总大小
    
    Args:
        directory: 目录路径
        
    Returns:
        目录大小（字节）
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        st.warning(f"⚠️ 无法计算目录大小：{str(e)}")
    return total_size


def safe_remove_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    安全地删除文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        (是否成功, 错误信息)
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True, None
        else:
            return False, "文件不存在"
    except Exception as e:
        return False, f"删除失败：{str(e)}"

