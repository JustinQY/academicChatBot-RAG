"""
RAG 系统核心模块
实现双向量库架构、文档索引、检索功能
"""

import os
from typing import List, Tuple, Optional
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document


def loadAndIndexFiles(
    file_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_type: str = "base",
    additional_metadata: Optional[dict] = None
) -> Tuple[List[Document], int]:
    """
    通用文档加载和索引函数
    
    用于加载PDF文件、分割文本并返回文档片段
    这个函数被基础向量库和用户向量库共同使用
    
    Args:
        file_paths: PDF 文件路径列表
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        source_type: 文档来源类型 ("base" 或 "user")
        additional_metadata: 额外的元数据（用于用户上传文档）
        
    Returns:
        (文档片段列表, 原始文档数量)
    """
    all_docs = []
    
    # 加载所有PDF文件
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # 添加元数据
            for doc in docs:
                doc.metadata['source_type'] = source_type
                if additional_metadata:
                    doc.metadata.update(additional_metadata)
            
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"❌ 加载文件失败 {file_path}: {str(e)}")
    
    if not all_docs:
        return [], 0
    
    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(all_docs)
    
    return splits, len(all_docs)


class DualVectorStoreRAG:
    """双向量库 RAG 系统"""
    
    def __init__(
        self,
        base_persist_dir: str = "./chroma_db/base",
        user_persist_dir: str = "./chroma_db/user",
        base_docs_dir: str = "CourseMaterials/deep_learning",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        初始化双向量库 RAG 系统
        
        Args:
            base_persist_dir: 基础向量库持久化目录
            user_persist_dir: 用户向量库持久化目录
            base_docs_dir: 基础文档目录
            embedding_model: OpenAI embedding 模型名称
        """
        self.base_persist_dir = base_persist_dir
        self.user_persist_dir = user_persist_dir
        self.base_docs_dir = base_docs_dir
        self.embedding_model = embedding_model
        
        # 创建目录
        os.makedirs(base_persist_dir, exist_ok=True)
        os.makedirs(user_persist_dir, exist_ok=True)
        
        # 初始化 embedding 函数
        self.embedding_function = OpenAIEmbeddings(model=embedding_model)
        
        # 初始化向量库
        self.base_vectorstore = None
        self.user_vectorstore = None
        self.base_doc_count = 0
        
    def initialize_base_vectorstore(self) -> int:
        """
        初始化或加载基础向量库
        
        Returns:
            加载的文档数量
        """
        # 检查是否已存在持久化的向量库
        if os.path.exists(self.base_persist_dir) and os.listdir(self.base_persist_dir):
            # 加载已有的向量库
            try:
                self.base_vectorstore = Chroma(
                    persist_directory=self.base_persist_dir,
                    embedding_function=self.embedding_function
                )
                # 尝试获取文档数量
                try:
                    collection = self.base_vectorstore._collection
                    self.base_doc_count = collection.count()
                except:
                    self.base_doc_count = 0
                
                return self.base_doc_count
            except Exception as e:
                st.warning(f"⚠️ 加载基础向量库失败，将重新创建：{str(e)}")
        
        # 首次创建：加载基础文档
        if not os.path.exists(self.base_docs_dir):
            st.error(f"❌ 基础文档目录不存在：{self.base_docs_dir}")
            return 0
        
        # 获取所有PDF文件
        pdf_files = []
        for root, dirs, files in os.walk(self.base_docs_dir):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            st.warning(f"⚠️ 在 {self.base_docs_dir} 中未找到 PDF 文件")
            return 0
        
        # 加载和索引文件
        splits, doc_count = loadAndIndexFiles(
            file_paths=pdf_files,
            source_type="base"
        )
        
        if not splits:
            st.error("❌ 未能加载任何基础文档")
            return 0
        
        # 创建向量库
        self.base_vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_function,
            persist_directory=self.base_persist_dir
        )
        
        self.base_doc_count = doc_count
        return doc_count
    
    def initialize_user_vectorstore(self):
        """初始化或加载用户向量库"""
        # 始终尝试加载用户向量库（可能为空）
        self.user_vectorstore = Chroma(
            persist_directory=self.user_persist_dir,
            embedding_function=self.embedding_function
        )
    
    def add_user_document(
        self,
        file_path: str,
        original_filename: str,
        upload_time: str,
        file_size: int
    ) -> Tuple[bool, str, int]:
        """
        添加用户上传的文档到用户向量库
        
        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            upload_time: 上传时间
            file_size: 文件大小
            
        Returns:
            (是否成功, 消息, 添加的文本块数量)
        """
        try:
            # 加载和索引文件
            additional_metadata = {
                'original_filename': original_filename,
                'upload_time': upload_time,
                'file_size': file_size
            }
            
            splits, doc_count = loadAndIndexFiles(
                file_paths=[file_path],
                source_type="user",
                additional_metadata=additional_metadata
            )
            
            if not splits:
                return False, "❌ 文档处理失败：未能提取任何内容", 0
            
            # 添加到用户向量库
            if self.user_vectorstore is None:
                self.initialize_user_vectorstore()
            
            self.user_vectorstore.add_documents(splits)
            
            return True, f"✅ 成功索引文档，添加了 {len(splits)} 个文本块", len(splits)
            
        except Exception as e:
            return False, f"❌ 索引文档时出错：{str(e)}", 0
    
    def remove_user_document(self, original_filename: str) -> Tuple[bool, str]:
        """
        从用户向量库中删除文档
        
        Args:
            original_filename: 原始文件名
            
        Returns:
            (是否成功, 消息)
        """
        try:
            if self.user_vectorstore is None:
                return True, "用户向量库为空"
            
            # 通过元数据过滤删除
            # 注意：Chroma 的删除操作需要文档ID，这里我们需要先查询
            collection = self.user_vectorstore._collection
            results = collection.get(
                where={"original_filename": original_filename}
            )
            
            if results and 'ids' in results and results['ids']:
                collection.delete(ids=results['ids'])
                return True, f"✅ 已从向量库中删除 {len(results['ids'])} 个文本块"
            else:
                return True, "向量库中未找到相关内容"
                
        except Exception as e:
            return False, f"⚠️ 从向量库删除时出错：{str(e)}"
    
    def create_rag_chain(self, k: int = 3):
        """
        创建 RAG 检索链
        
        Args:
            k: 检索的文档数量
            
        Returns:
            RAG chain
        """
        # 创建混合检索器
        def hybrid_retrieve(query: str) -> List[Document]:
            """从两个向量库中检索相关文档"""
            all_docs = []
            
            # 从基础库检索
            if self.base_vectorstore:
                try:
                    base_docs = self.base_vectorstore.similarity_search(query, k=k)
                    all_docs.extend(base_docs)
                except Exception as e:
                    st.warning(f"⚠️ 基础库检索失败：{str(e)}")
            
            # 从用户库检索
            if self.user_vectorstore:
                try:
                    # 检查用户库是否有内容
                    collection = self.user_vectorstore._collection
                    if collection.count() > 0:
                        user_docs = self.user_vectorstore.similarity_search(query, k=k)
                        all_docs.extend(user_docs)
                except Exception as e:
                    # 用户库可能为空，这是正常的
                    pass
            
            # 返回前 k 个文档（可以添加重新排序逻辑）
            return all_docs[:k]
        
        # 格式化文档，添加来源标记
        def format_docs_with_source(docs: List[Document]) -> str:
            """格式化文档并标记来源"""
            parts = []
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown_source")
                src_type = d.metadata.get("source_type", "base")
                page = d.metadata.get("page_label", d.metadata.get("page", "unknown_page"))
                
                # 根据来源类型选择图标
                if src_type == "user":
                    emoji = "📄"
                    original_name = d.metadata.get("original_filename", "Unknown")
                    upload_time = d.metadata.get("upload_time", "Unknown")
                    header = f"{emoji} [{i}] 用户文档：{original_name} (上传于 {upload_time}, p.{page})"
                else:
                    emoji = "📘"
                    header = f"{emoji} [{i}] 课程材料：{os.path.basename(src)}, p.{page}"
                
                text = (d.page_content or "").strip()
                parts.append(f"{header}\n{text}")
            
            return "\n\n".join(parts)
        
        # 构建 RAG 链
        prompt_template = """You are a helpful assistant.
Answer the question using ONLY the Context below.
If the answer is not in the Context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        rag_chain = (
            {
                "context": RunnableLambda(hybrid_retrieve) | RunnableLambda(format_docs_with_source),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        
        return rag_chain

