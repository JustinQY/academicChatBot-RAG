"""
文档管理模块
负责文档的上传、存储、删除、元数据管理
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import streamlit as st
from utils import (
    generate_unique_filename, 
    calculate_file_hash, 
    validate_pdf_file,
    safe_remove_file,
    format_file_size
)


class DocumentManager:
    """文档管理器类"""
    
    def __init__(self, upload_dir: str = "UserUploads", metadata_file: str = "document_metadata.json"):
        """
        初始化文档管理器
        
        Args:
            upload_dir: 上传文件存储目录
            metadata_file: 元数据文件名
        """
        self.upload_dir = upload_dir
        self.metadata_file = os.path.join(upload_dir, metadata_file)
        self._ensure_directory_exists()
        
    def _ensure_directory_exists(self):
        """确保上传目录存在"""
        os.makedirs(self.upload_dir, exist_ok=True)
        
    def _load_metadata(self) -> Dict[str, Dict]:
        """
        加载文档元数据
        
        Returns:
            文档元数据字典 {file_id: metadata}
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"⚠️ 无法加载元数据：{str(e)}")
                return {}
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Dict]):
        """
        保存文档元数据
        
        Args:
            metadata: 文档元数据字典
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"❌ 无法保存元数据：{str(e)}")
    
    def check_duplicate(self, file_content: bytes) -> Optional[Dict]:
        """
        检查文件是否已存在（通过哈希值）
        
        Args:
            file_content: 文件内容
            
        Returns:
            如果存在重复，返回已存在文件的元数据；否则返回 None
        """
        file_hash = calculate_file_hash(file_content)
        metadata = self._load_metadata()
        
        for file_id, meta in metadata.items():
            if meta.get('hash') == file_hash:
                return meta
        return None
    
    def upload_document(self, uploaded_file) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        上传并保存文档
        
        Args:
            uploaded_file: Streamlit UploadedFile 对象
            
        Returns:
            (是否成功, 错误信息/成功信息, 文档元数据)
        """
        # 1. 验证文件
        is_valid, error_msg = validate_pdf_file(uploaded_file)
        if not is_valid:
            return False, error_msg, None
        
        # 2. 读取文件内容
        try:
            file_content = uploaded_file.getvalue()
        except Exception as e:
            return False, f"❌ 无法读取文件：{str(e)}", None
        
        # 3. 检查重复
        duplicate = self.check_duplicate(file_content)
        if duplicate:
            return False, f"⚠️ 文件已存在！\n文件名：{duplicate['original_filename']}\n上传时间：{duplicate['upload_time']}", None
        
        # 4. 生成唯一文件名
        unique_filename = generate_unique_filename(uploaded_file.name)
        filepath = os.path.join(self.upload_dir, unique_filename)
        
        # 5. 保存文件
        try:
            with open(filepath, 'wb') as f:
                f.write(file_content)
        except Exception as e:
            return False, f"❌ 保存文件失败：{str(e)}", None
        
        # 6. 创建元数据
        file_hash = calculate_file_hash(file_content)
        metadata_entry = {
            'file_id': unique_filename,
            'original_filename': uploaded_file.name,
            'filepath': filepath,
            'size': len(file_content),
            'size_formatted': format_file_size(len(file_content)),
            'hash': file_hash,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'indexed': False  # 标记是否已索引到向量库
        }
        
        # 7. 保存元数据
        all_metadata = self._load_metadata()
        all_metadata[unique_filename] = metadata_entry
        self._save_metadata(all_metadata)
        
        return True, f"✅ 文件上传成功：{uploaded_file.name}", metadata_entry
    
    def delete_document(self, file_id: str) -> Tuple[bool, Optional[str]]:
        """
        删除文档及其元数据
        
        Args:
            file_id: 文件ID（唯一文件名）
            
        Returns:
            (是否成功, 错误信息/成功信息)
        """
        # 1. 加载元数据
        all_metadata = self._load_metadata()
        
        if file_id not in all_metadata:
            return False, "❌ 文件不存在！"
        
        # 2. 删除物理文件
        filepath = all_metadata[file_id]['filepath']
        success, error = safe_remove_file(filepath)
        
        if not success:
            return False, f"❌ 删除文件失败：{error}"
        
        # 3. 删除元数据
        original_name = all_metadata[file_id]['original_filename']
        del all_metadata[file_id]
        self._save_metadata(all_metadata)
        
        return True, f"✅ 已删除文档：{original_name}"
    
    def list_documents(self) -> List[Dict]:
        """
        列出所有已上传的文档
        
        Returns:
            文档列表（按上传时间倒序）
        """
        all_metadata = self._load_metadata()
        documents = list(all_metadata.values())
        
        # 按上传时间倒序排序
        documents.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return documents
    
    def mark_as_indexed(self, file_id: str):
        """
        标记文档为已索引
        
        Args:
            file_id: 文件ID
        """
        all_metadata = self._load_metadata()
        if file_id in all_metadata:
            all_metadata[file_id]['indexed'] = True
            self._save_metadata(all_metadata)
    
    def get_document_metadata(self, file_id: str) -> Optional[Dict]:
        """
        获取指定文档的元数据
        
        Args:
            file_id: 文件ID
            
        Returns:
            文档元数据，如果不存在返回 None
        """
        all_metadata = self._load_metadata()
        return all_metadata.get(file_id)

