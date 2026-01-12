"""
学术聊天机器人 - 主应用
支持基础课程材料问答 + 用户文档上传
"""

import streamlit as st
import os
import json
from datetime import datetime
from document_manager import DocumentManager
from rag_system import DualVectorStoreRAG
from utils import format_file_size, get_directory_size, safe_remove_file

# 页面配置
st.set_page_config(
    page_title="学术聊天机器人", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📑 McMaster Academic Knowledge QA System V1.0")
st.markdown("Based on course materials default and uploaded by users and RAG.")


# ==================== 配置加载 ====================
@st.cache_resource
def load_config():
    """
    配置加载优先级：
    1. Streamlit Secrets（推荐用于部署）
    2. 环境变量
    3. config.json 文件（本地开发）
    """
    # 获取 OpenAI API Key
    openai_key = None
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        openai_key = st.secrets['OPENAI_API_KEY']
        source = "Streamlit Secrets"
    elif 'OPENAI_API_KEY' in os.environ:
        openai_key = os.environ['OPENAI_API_KEY']
        source = "Environment Variable"
    else:
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            openai_key = config.get("OpenAIAPIKey")
            source = "config.json"
        except FileNotFoundError:
            pass
    
    if not openai_key:
        st.error("""
        ❌ 未找到 OpenAI API Key！
        
        请通过以下任一方式配置：
        
        **1. Streamlit Cloud 部署（推荐）：**
        - 在 Streamlit Cloud 设置中添加 Secrets
        - 格式: `OPENAI_API_KEY = "your-key-here"`
        
        **2. 本地环境变量：**
        ```bash
        export OPENAI_API_KEY="your-key-here"
        ```
        
        **3. 本地 config.json 文件：**
        ```json
        {
          "OpenAIAPIKey": "your-key-here"
        }
        ```
        """)
        st.stop()
    
    # 设置 OpenAI API Key
    os.environ['OPENAI_API_KEY'] = openai_key
    
    # 获取 LangChain API Key (可选)
    langchain_key = None
    if hasattr(st, 'secrets') and 'LANGCHAIN_API_KEY' in st.secrets:
        langchain_key = st.secrets['LANGCHAIN_API_KEY']
    elif 'LANGCHAIN_API_KEY' in os.environ:
        langchain_key = os.environ['LANGCHAIN_API_KEY']
    else:
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            langchain_key = config.get("LangChainAPIKey")
        except:
            pass
    
    # 配置 LangSmith 追踪（如果提供了 API Key）
    if langchain_key:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        os.environ['LANGCHAIN_API_KEY'] = langchain_key
    
    return {
        'source': source,
        'langsmith_enabled': bool(langchain_key)
    }


# ==================== 初始化 RAG 系统 ====================
@st.cache_resource
def initialize_rag_system():
    """初始化双向量库 RAG 系统（基础库缓存）"""
    rag = DualVectorStoreRAG()
    
    # 初始化基础向量库（缓存）
    with st.spinner("📚 正在初始化基础知识库..."):
        base_doc_count = rag.initialize_base_vectorstore()
    
    # 初始化用户向量库（不缓存，动态）
    rag.initialize_user_vectorstore()
    
    return rag, base_doc_count


# ==================== 初始化文档管理器 ====================
def get_document_manager():
    """获取文档管理器实例"""
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    return st.session_state.doc_manager


# ==================== 主应用逻辑 ====================
def main():
    try:
        # 加载配置
        config = load_config()
        
        # 初始化 RAG 系统
        rag_system, base_doc_count = initialize_rag_system()
        st.success(f"✌️ System All Set!  {base_doc_count} default docs loaded!")
        
        # 初始化文档管理器
        doc_manager = get_document_manager()
        
        # 初始化会话状态
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        if 'show_doc_manager' not in st.session_state:
            st.session_state.show_doc_manager = False
        
        # ==================== 文档上传区域 ====================
        st.markdown("---")
        st.markdown("### 🛜 Upload Your Documents Here")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload your PDF files to library (batch uploading available)",
                type=['pdf'],
                accept_multiple_files=True,
                help="PDF files only, allowed to select multiple files with a 50 MB size limit for each one.",
                key="pdf_uploader"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # 对齐按钮
            if st.button("Manage Uploaded Docs", use_container_width=True):
                st.session_state.show_doc_manager = not st.session_state.show_doc_manager
        
        # ==================== 批量文件上传处理 ====================
        if uploaded_files is not None and len(uploaded_files) > 0:
            # 导入批量上传辅助模块
            from batch_upload_helper import (
                generate_batch_id, initialize_batch_state, get_file_key,
                update_file_status, get_pending_files, get_failed_files,
                get_batch_progress, get_batch_summary, should_process_file
            )
            
            # 生成当前批次ID
            current_batch_id = generate_batch_id(uploaded_files)
            
            # 初始化批次状态（如果是新批次）
            if 'batch_upload_state' not in st.session_state:
                st.session_state.batch_upload_state = None
            
            # 检查是否是新批次
            if (st.session_state.batch_upload_state is None or 
                st.session_state.batch_upload_state['batch_id'] != current_batch_id):
                # 新批次，初始化状态
                st.session_state.batch_upload_state = initialize_batch_state(
                    uploaded_files, current_batch_id
                )
            
            batch_state = st.session_state.batch_upload_state
            
            # 显示文件选择信息
            if len(uploaded_files) > 1:
                st.info(f"📦 已选择 {len(uploaded_files)} 个文件")
            
            # 显示整体进度
            progress = get_batch_progress(batch_state)
            if progress > 0:
                st.progress(progress, text=get_batch_summary(batch_state))
            
            # 处理待处理和失败的文件
            files_to_process = []
            for file in uploaded_files:
                file_key = get_file_key(file)
                if should_process_file(batch_state, file_key):
                    files_to_process.append((file, file_key))
            
            # 如果有文件需要处理
            if files_to_process:
                # 顺序处理每个文件
                for file, file_key in files_to_process:
                    file_info = batch_state['files'][file_key]
                    
                    # 标记为处理中
                    update_file_status(batch_state, file_key, 'processing')
                    
                    with st.status(f"📥 正在处理: {file_info['filename']}", expanded=True) as status:
                        try:
                            # 阶段1: 上传和保存文件
                            st.write("验证文件格式和大小...")
                            success, message, metadata = doc_manager.upload_document(file)
                            
                            if not success:
                                # 上传失败（格式错误、重复文件等）
                                status.update(label=f"❌ {file_info['filename']} 上传失败", state="error")
                                update_file_status(batch_state, file_key, 'failed', error=message)
                                st.error(f"❌ {message}")
                                continue  # 跳过这个文件，继续下一个
                            
                            st.write("✅ 文件保存成功")
                            
                            # 阶段2: 索引到向量库
                            st.write("🔢 正在向量化文档...")
                            index_success, index_message, chunk_count = rag_system.add_user_document(
                                file_path=metadata['filepath'],
                                original_filename=metadata['original_filename'],
                                upload_time=metadata['upload_time'],
                                file_size=metadata['size']
                            )
                            
                            if not index_success:
                                # 索引失败，清理已保存的文件
                                status.update(label=f"❌ {file_info['filename']} 索引失败", state="error")
                                st.error(index_message)
                                st.warning("正在清理已保存的文件...")
                                
                                file_success, file_error = safe_remove_file(metadata['filepath'])
                                if file_success:
                                    st.info("✅ 已清理失败的上传")
                                
                                update_file_status(batch_state, file_key, 'failed', error=index_message)
                                continue  # 跳过这个文件，继续下一个
                            
                            # 阶段3: 保存元数据
                            save_success, save_error = doc_manager.save_document_metadata(metadata)
                            
                            if not save_success:
                                # 元数据保存失败（极少见）
                                status.update(label=f"⚠️ {file_info['filename']} 元数据保存失败", state="error")
                                st.error(f"❌ {save_error}")
                                st.warning("文档已索引但元数据未保存，可能导致重复上传检测失败")
                                update_file_status(batch_state, file_key, 'failed', error=save_error)
                                continue
                            
                            # 全部成功
                            status.update(label=f"✅ {file_info['filename']} 处理完成", state="complete")
                            st.success(f"🎉 {metadata['original_filename']} 已成功添加到知识库！")
                            st.info(index_message)
                            
                            update_file_status(batch_state, file_key, 'success')
                            
                        except Exception as e:
                            # 意外错误
                            status.update(label=f"❌ {file_info['filename']} 处理异常", state="error")
                            error_msg = f"处理文件时发生错误: {str(e)}"
                            st.error(error_msg)
                            update_file_status(batch_state, file_key, 'failed', error=error_msg)
                    
                    # 每处理完一个文件，刷新一次页面以更新进度
                    if batch_state['completed_files'] < batch_state['total_files']:
                        st.rerun()
            
            # 批次处理完成
            if batch_state['overall_status'] == 'completed':
                st.markdown("---")
                
                # 显示批次摘要
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ 成功", batch_state['success_count'])
                with col2:
                    st.metric("❌ 失败", batch_state['failed_count'])
                with col3:
                    st.metric("📊 总计", batch_state['total_files'])
                
                # 显示失败文件详情
                if batch_state['failed_count'] > 0:
                    with st.expander("查看失败文件详情", expanded=False):
                        for file_key, file_info in batch_state['files'].items():
                            if file_info['status'] == 'failed':
                                st.error(f"**{file_info['filename']}**: {file_info['error']}")
                    
                    # 提供重试选项
                    if st.button("🔄 重试失败的文件"):
                        # 将失败文件重置为 pending 状态
                        for file_key in get_failed_files(batch_state):
                            batch_state['files'][file_key]['status'] = 'pending'
                            batch_state['files'][file_key]['error'] = None
                        
                        batch_state['overall_status'] = 'processing'
                        batch_state['failed_count'] = 0
                        batch_state['completed_files'] -= len(get_failed_files(batch_state))
                        st.rerun()
                
                # 完成后的提示
                if batch_state['success_count'] > 0:
                    st.success(f"🎊 批量上传完成！成功处理 {batch_state['success_count']} 个文件")
        
        # ==================== 文档管理浮窗 ====================
        if st.session_state.show_doc_manager:
            with st.expander("📚 已上传文档管理", expanded=True):
                documents = doc_manager.list_documents()
                
                if not documents:
                    st.info("📭 还没有上传任何文档")
                else:
                    st.caption(f"共 {len(documents)} 个文档")
                    
                    for doc in documents:
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            st.markdown(f"**📄 {doc['original_filename']}**")
                        
                        with col2:
                            st.text(f"📦 {doc['size_formatted']}")
                        
                        with col3:
                            st.text(f"🕐 {doc['upload_time']}")
                        
                        with col4:
                            if st.button("🗑️", key=f"del_{doc['file_id']}", help="Delete"):
                                # 删除文件
                                file_success, file_message = doc_manager.delete_document(doc['file_id'])
                                
                                # 从向量库删除
                                vec_success, vec_message = rag_system.remove_user_document(
                                    doc['original_filename']
                                )
                                
                                if file_success:
                                    st.success(file_message)
                                    if vec_success:
                                        st.info(vec_message)
                                    else:
                                        st.warning(vec_message)
                                    st.rerun()
                                else:
                                    st.error(file_message)
                        
                        st.markdown("---")
        
        # ==================== 问答区域 ====================
        st.markdown("---")
        st.markdown("### 🙋 Question")
        
        question = st.text_area(
            f"Please enter your question here: \n (**Note: The system currently answers questions only based on materials in the database, it'll answer *I don't know based on the provided context.* if it failed to find answer from the docs.**)",
            placeholder="Can you list some of the hyperparameters in the FFN?",
            height=100,
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            ask_button = st.button("Shoot", type="primary", use_container_width=True)
        with col2:
            if st.button("Clean The History", use_container_width=True):
                st.session_state.qa_history = []
                st.rerun()
        
        if ask_button and question.strip():
            with st.spinner("(ー_ーゞ thinking~~~"):
                try:
                    # 创建 RAG 链并查询
                    rag_chain = rag_system.create_rag_chain(k=3)
                    response = rag_chain.invoke(question)
                    
                    # 保存到历史记录
                    qa_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'question': question.strip(),
                        'answer': response.content
                    }
                    st.session_state.qa_history.append(qa_entry)
                    
                    # 显示当前回答
                    st.markdown("### Answer")
                    st.info(response.content)
                    
                except Exception as e:
                    st.error(f"😭 Get an error: {str(e)}")
        
        elif ask_button:
            st.warning("🤔 Got nothing to ask yet?")
        
        # ==================== 问答历史记录 ====================
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("## QA History")
            st.caption(f"You got {len(st.session_state.qa_history)} histories in total.")
            
            # 逆序显示（最新的在上面）
            for idx, qa in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(
                    f"🕐 {qa['timestamp']} - Question #{len(st.session_state.qa_history) - idx + 1}",
                    expanded=(idx == 1)
                ):
                    st.markdown(f"**Question:**")
                    st.write(qa['question'])
                    st.markdown(f"**Answer:**")
                    st.info(qa['answer'])
        
        # ==================== 侧边栏 ====================
        with st.sidebar:
            st.header("About")
            st.markdown("""
            This is a academic QA system on the strength of RAG.
            
            **Highlights:**
            - Supports users to upload custom docs.
            - Load, analyze and process pdf type of docs.
            - Performs semantic retrieval to identify the most relevant content chunks.
            - Generates accurate answers using OpenAI GPT-3.5 within a RAG pipeline.
            - Implements a LangChain-based Retrieval-Augmented Generation workflow.
            - Restricts responses only based on course materials to minimize hallucinations.
            - Automatically stores QA history for session continuity.
            - Clearly distinguishes content sources (default vs. user-uploaded).
            
            **Guide:**
            1. Upload your PDF docs (if you'd like to ask questions related to them).
            2. Enter your question in the input field.
            3. Click the “Shoot” button.
            4. Wait for the answer.
            5. Previous questions and answers will be automatically saved below.
            6. Click “Manage Uploaded Docs” to view or remove uploaded files.

            **Sample Questions: **
            - Can you list some of the hyperparameters in the FFN?
            - What is backpropagation?
            - Explain gradient descent
            """)
            
            st.divider()
            
            st.header("🧑‍💻 Tech Stack:")
            st.markdown("""
            - **Front End**: Streamlit by Codegen
            - **LLM**: OpenAI GPT-3.5
            - **Vector Database**: Chroma (Locally Persistence)
            - **Framework**: LangChain
            - **Docs Processing**: PyPDF
            - **Architecture**: Double Vector Database (Default + User)
            """)
            
            st.divider()
            
            # 存储使用情况
            try:
                upload_dir = "UserUploads"
                if os.path.exists(upload_dir):
                    total_size = get_directory_size(upload_dir)
                    st.metric(
                        label="📊 Data Uploaded",
                        value=format_file_size(total_size)
                    )
            except:
                pass
            
            st.markdown("---")
            st.caption("🔔 Note: Initial document loading and vectorization may take a few moments on first use.")
    
    except Exception as e:
        st.error(f"❌ 系统错误：{str(e)}")
        st.info("""
        请检查：
        - OpenAI API Key 是否正确
        - CourseMaterials/deep_learning 目录下是否有PDF文件
        - 网络连接是否正常
        """)


if __name__ == "__main__":
    main()
