import streamlit as st
import os, json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 页面配置
st.set_page_config(page_title="学术聊天机器人", page_icon="🎓", layout="wide")
st.title("🎓 学术课程问答助手")
st.markdown("基于深度学习课程材料的RAG问答系统")

# 加载配置
@st.cache_resource
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        os.environ['OPENAI_API_KEY'] = config["OpenAIAPIKey"]
        if "LangChainAPIKey" in config:
            os.environ['LANGCHAIN_API_KEY'] = config["LangChainAPIKey"]
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        return config
    except FileNotFoundError:
        st.error("❌ 找不到 config.json 文件，请根据 config.example.json 创建配置文件！")
        st.stop()
    except KeyError as e:
        st.error(f"❌ 配置文件缺少必要的键: {e}")
        st.stop()

# 初始化RAG系统
@st.cache_resource
def initialize_rag():
    try:
        # 加载文档
        loader = DirectoryLoader(
            "CourseMaterials/deep_learning",
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        
        if not docs:
            st.error("❌ 没有找到PDF文档，请确保 CourseMaterials/deep_learning 目录下有PDF文件！")
            st.stop()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300,
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # 向量化
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
        # 构建RAG链
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
        
        def format_docs(docs):
            parts = []
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown_source")
                page = d.metadata.get("page_label", d.metadata.get("page", "unknown_page"))
                text = (d.page_content or "").strip()
                parts.append(f"[{i}] ({src}, p.{page})\n{text}")
            return "\n\n".join(parts)
        
        rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        
        return rag_chain, len(docs)
    except Exception as e:
        st.error(f"❌ 初始化RAG系统时出错: {str(e)}")
        st.stop()

# 主界面
try:
    config = load_config()
    with st.spinner("🔄 正在初始化RAG系统，首次加载可能需要一些时间..."):
        rag_chain, doc_count = initialize_rag()
    st.success(f"✅ 系统已就绪！已加载 {doc_count} 个文档。")
    
    # 用户输入
    question = st.text_area(
        "💬 请输入你的问题：",
        placeholder="例如: Can you list some of the hyperparameters in the FFN?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 提问", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ 清除", use_container_width=True):
            st.rerun()
    
    if ask_button:
        if question.strip():
            with st.spinner("🤔 正在思考中..."):
                try:
                    response = rag_chain.invoke(question)
                    
                    # 显示回答
                    st.markdown("### 📝 回答：")
                    st.info(response.content)
                    
                except Exception as e:
                    st.error(f"❌ 生成回答时出错: {str(e)}")
        else:
            st.warning("⚠️ 请先输入问题")
            
    # 侧边栏
    with st.sidebar:
        st.header("📚 关于系统")
        st.markdown("""
        这是一个基于RAG（检索增强生成）的学术问答系统。
        
        **功能特点：**
        - 📖 自动读取深度学习课程PDF文档
        - 🔍 智能检索相关内容片段
        - 💡 基于OpenAI GPT-3.5生成准确答案
        - ⚡ 使用LangChain构建RAG流程
        - 🎯 仅基于课程材料回答，避免虚构信息
        
        **使用说明：**
        1. 在输入框中输入你的问题
        2. 点击"提问"按钮
        3. 等待系统检索并生成答案
        
        **示例问题：**
        - Can you list some of the hyperparameters in the FFN?
        - What is backpropagation?
        - Explain gradient descent
        """)
        
        st.divider()
        
        st.header("⚙️ 技术栈")
        st.markdown("""
        - **前端**: Streamlit
        - **LLM**: OpenAI GPT-3.5
        - **向量数据库**: Chroma
        - **框架**: LangChain
        - **文档处理**: PyPDF
        """)
        
        st.divider()
        
        st.markdown("---")
        st.caption("💡 提示：首次使用时系统会加载所有PDF文档并进行向量化，可能需要几分钟时间。之后使用Streamlit缓存会加快响应速度。")
        
except Exception as e:
    st.error(f"❌ 系统错误: {str(e)}")
    st.info("请检查：\n- config.json 文件是否存在\n- OpenAI API Key 是否正确\n- CourseMaterials/deep_learning 目录下是否有PDF文件")

