"""
Insurance AI Agent - 主入口
基于 LangGraph + Streamlit 的保险智能助手 Web 界面。

启动方式：
    streamlit run app.py
"""

import sys
import os
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 Python 路径
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import glob as glob_module
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    LLM_CONFIG,
    EMBEDDING_CONFIG,
    STREAMLIT_CONFIG,
    PDF_CONFIG,
    RAG_ENGINE,
    VECTOR_STORE_CONFIG,
)
from graph.graph_builder import AgentGraphBuilder
from memory.state_manager import StateManager
from services.knowledge_service import KnowledgeService
from utils.logger import get_logger

logger = get_logger("app")

# ------------------------------------------------------------
# 页面配置
# ------------------------------------------------------------
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
)

# ------------------------------------------------------------
# 全局服务初始化（缓存，避免重复创建）
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_services(rag_engine: str = "langchain") -> Dict[str, Any]:
    """
    初始化所有服务：知识库服务、Agent 图构建器、状态管理器。

    Args:
        rag_engine: RAG 引擎选择（"langchain" | "llamaindex"）

    Returns:
        包含各服务实例的字典
    """
    # 将引擎设置写入环境变量（确保子模块读取一致）
    os.environ["RAG_ENGINE"] = rag_engine

    knowledge_service = KnowledgeService()

    # 自动加载或构建知识库
    if not knowledge_service.get_stats().get("index_exists"):
        knowledge_service.build_knowledge_base()
    else:
        knowledge_service.load_existing_index()

    graph_builder = AgentGraphBuilder()
    state_manager = StateManager(graph_builder.checkpointer)

    return {
        "knowledge_service": knowledge_service,
        "graph_builder": graph_builder,
        "state_manager": state_manager,
    }


# 跟踪当前 RAG 引擎（用于热切换）
if "_current_rag_engine" not in st.session_state:
    st.session_state._current_rag_engine = RAG_ENGINE

# 初始化
with st.spinner("正在初始化系统..."):
    services = init_services(rag_engine=st.session_state._current_rag_engine)

knowledge_service: KnowledgeService = services["knowledge_service"]
graph_builder: AgentGraphBuilder = services["graph_builder"]
state_manager: StateManager = services["state_manager"]

# ------------------------------------------------------------
# 多线程会话管理：sessions 字典 + session_id 指针
# （必须在侧边栏之前初始化，因为侧边栏会读取 sessions）
# ------------------------------------------------------------
if "sessions" not in st.session_state:
    default_id = f"session_{int(time.time())}"
    st.session_state.sessions = {
        default_id: {
            "id": default_id,
            "messages": [],
            "created_at": time.strftime("%H:%M:%S"),
            "preview": "对话 1",
            "_number": 1,
        }
    }
    st.session_state.session_id = default_id

# ------------------------------------------------------------
# 侧边栏：知识库管理 + 线程管理
# ------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Insurance AI Agent")
    st.divider()

    # --- 线程管理 ---
    st.subheader("🧵 线程管理")
    num_sessions = len(st.session_state.sessions)
    st.caption(f"共 {num_sessions} 个对话线程，在顶部下拉框切换")
    for tid in list(st.session_state.sessions.keys()):
        sess = st.session_state.sessions[tid]
        is_active = tid == st.session_state.session_id
        prefix = "🟢 " if is_active else "⚪ "
        num = sess.get("_number", "?")
        st.text(f"{prefix}#{num} {sess['preview'][:16]}")

    st.divider()

    # --- 知识库状态 ---
    st.subheader("📚 Knowledge Base")
    kb_stats: Dict[str, Any] = knowledge_service.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDF 数量", kb_stats.get("pdf_count", 0))
    with col2:
        st.metric("向量数量", kb_stats.get("total_vectors", 0))

    with st.expander("查看详情"):
        st.markdown(f"""
        - **Embedding 模型**: {kb_stats.get('embedding_model', 'N/A')}
        - **Chunk Size**: {kb_stats.get('chunk_size', 'N/A')}
        - **Chunk Overlap**: {kb_stats.get('chunk_overlap', 'N/A')}
        - **索引状态**: {'已加载' if kb_stats.get('index_loaded') else '未加载'}
        - **LLM 模型**: {LLM_CONFIG['model']}
        """)

    # --- RAG 引擎切换 ---
    st.divider()
    st.subheader("🔧 RAG 引擎")
    rag_engine_options = ["langchain", "llamaindex"]
    current_idx = 0 if st.session_state._current_rag_engine == "langchain" else 1
    selected_engine = st.selectbox(
        "选择 RAG 引擎",
        options=rag_engine_options,
        index=current_idx,
        help="LangChain: PyMuPDF + RecursiveCharacterTextSplitter + FAISS\n"
             "LlamaIndex: SimpleDirectoryReader + SentenceSplitter + VectorStoreIndex + FAISS",
    )
    if selected_engine != st.session_state._current_rag_engine:
        st.session_state._current_rag_engine = selected_engine
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # --- 管理员功能 ---
    st.subheader("⚙️ 管理员")

    # 上传 PDF
    uploaded_files = st.file_uploader(
        "上传 PDF 文档",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传保险条款 PDF 文档到知识库",
    )
    if uploaded_files:
        if st.button("📤 上传并处理", use_container_width=True):
            saved_count: int = 0
            for uploaded_file in uploaded_files:
                save_path: Path = Path(PDF_CONFIG["pdf_dir"]) / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1

            # 重建索引
            with st.spinner("正在重建知识库索引..."):
                result = knowledge_service.rebuild()
            if result["success"]:
                st.success(f"已上传 {saved_count} 个 PDF，{result['message']}")
                st.rerun()
            else:
                st.error(result["message"])

    # 管理按钮
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 重建索引", use_container_width=True):
            with st.spinner("正在重建知识库..."):
                result = knowledge_service.rebuild()
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])

    with col_b:
        if st.button("🗑️ 删除知识库", use_container_width=True):
            if st.warning("确定删除知识库索引？"):
                result = knowledge_service.delete_knowledge_base()
                st.success(result["message"])
                st.rerun()

    # PDF 文件列表
    st.divider()
    st.subheader("📄 PDF 文件列表")
    pdf_files: List[Dict] = knowledge_service.get_pdf_files()
    if pdf_files:
        for f in pdf_files:
            st.text(f"📎 {f['name']} ({f['size']:,} bytes)")
    else:
        st.caption("暂无 PDF 文件")

    st.divider()

    # --- 日志查看器 ---
    st.subheader("📋 日志查看")
    with st.expander("查看最近日志"):
        log_dir: Path = PROJECT_ROOT / "logs"
        log_files: list = sorted(
            log_dir.glob("agent_*.log"), reverse=True
        ) if log_dir.exists() else []

        if log_files:
            selected_log: str = st.selectbox(
                "选择日志文件",
                [f.name for f in log_files[:7]],
                label_visibility="collapsed",
            )
            if selected_log:
                log_path: Path = log_dir / selected_log
                try:
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines: list = lf.readlines()[-50:]  # 最近50行
                    st.code("".join(reversed(lines)), language="log")
                except Exception:
                    st.caption("无法读取日志文件")
        else:
            st.caption("暂无日志文件")

    st.divider()

    # --- 开发者调试面板 ---
    with st.expander("🛠️ Developer Panel"):
        kb_stats = knowledge_service.get_stats()
        retriever = knowledge_service.get_retriever()
        st.markdown(f"""
        - **Current RAG Engine**: {st.session_state._current_rag_engine}
        - **Embedding Model**: {kb_stats.get('embedding_model', 'N/A')}
        - **Vector Store**: FAISS
        - **Retriever Type**: {type(retriever).__name__}
        - **Chunk Size**: {kb_stats.get('chunk_size', 'N/A')}
        - **Chunk Overlap**: {kb_stats.get('chunk_overlap', 'N/A')}
        - **Top-K**: {VECTOR_STORE_CONFIG['top_k']}
        - **PDF Count**: {kb_stats.get('pdf_count', 0)}
        - **Total Vectors**: {kb_stats.get('total_vectors', 0)}
        """)

    # 关于
    with st.expander("ℹ️ 关于"):
        st.markdown("""
        **Insurance AI Agent**

        基于 LangGraph 构建的企业级保险智能助手。

        - Agent + Tool Calling + RAG
        - DeepSeek LLM
        - BAAI/bge-base-zh-v1.5 Embedding
        - FAISS 向量数据库
        """)

# ------------------------------------------------------------
# 主界面：聊天窗口
# ------------------------------------------------------------
st.title("🛡️ Insurance AI Agent")
st.caption("保险智能助手 — 您的专业保险顾问")

# ------------------------------------------------------------
# 主界面顶部：🧵 线程切换器
# ------------------------------------------------------------
st.info("🔍 DEBUG: 线程切换器代码段开始执行 — 如果看到这条消息说明代码已到达")

# 当前线程列表（按创建顺序排列，分配序号）
sids = list(st.session_state.sessions.keys())
# 给每个 session 分配一个稳定的序号
for i, tid in enumerate(sids, 1):
    st.session_state.sessions[tid]["_number"] = i

labels = [
    f"#{st.session_state.sessions[t].get('_number', '?')} — {st.session_state.sessions[t]['preview'][:16]}"
    for t in sids
]
cur = sids.index(st.session_state.session_id) if st.session_state.session_id in sids else 0

picked = st.selectbox(
    "🧵 切换对话线程",
    options=range(len(sids)),
    format_func=lambda i: labels[i],
    index=cur,
    key="thread_selector_v2",
)

# 切换线程后 Streamlit 自动 rerun，无需手动 rerun
if sids[picked] != st.session_state.session_id:
    st.session_state.session_id = sids[picked]

if st.button("➕ 新建对话线程", key="new_thread_btn"):
    nid = f"sess_{int(time.time())}"
    next_num = len(st.session_state.sessions) + 1
    st.session_state.sessions[nid] = {
        "id": nid,
        "messages": [],
        "created_at": time.strftime("%H:%M:%S"),
        "preview": f"对话 {next_num}",
        "_number": next_num,
    }
    st.session_state.session_id = nid
    st.rerun()

st.divider()

# 渲染当前线程的历史消息
_current_messages = st.session_state.sessions[st.session_state.session_id]["messages"]
for msg in _current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 显示工具调用记录
        if msg.get("tool_calls"):
            with st.expander("🔧 查看工具调用"):
                for tc in msg["tool_calls"]:
                    st.json(tc)

        # 显示 RAG 检索来源（增强版：区分引擎）
        if msg.get("retrieved_sources"):
            with st.expander("📖 查看引用来源"):
                for i, source in enumerate(msg["retrieved_sources"], 1):
                    engine = source.get("engine", "LangChain")
                    st.markdown(f"**来源 {i}** ({engine})")
                    st.caption(source.get("content", "")[:500])
                    st.divider()

        # 显示 Agent Monitor
        if msg.get("agent_monitor"):
            with st.expander("📊 Agent Monitor"):
                for entry in msg["agent_monitor"]:
                    step: str = entry.get("step", "")
                    if step == "agent":
                        st.info(f"""
                        **Agent 决策**
                        - 意图: {entry.get('intent', 'N/A')}
                        - 工具调用数: {entry.get('tool_calls_count', 0)}
                        - LLM 耗时: {entry.get('llm_time', 0):.2f}s
                        """)
                    elif step == "tool":
                        icon: str = "✅" if entry.get("success") else "❌"
                        st.success(f"""
                        {icon} **工具执行: {entry.get('tool_name', 'N/A')}**
                        - 参数: {entry.get('tool_args', {})}
                        - 耗时: {entry.get('tool_time', 0):.4f}s
                        """)


# ------------------------------------------------------------
# 聊天输入
# ------------------------------------------------------------
if prompt := st.chat_input("请输入您的保险相关问题..."):
    # 添加用户消息
    _cur_messages = st.session_state.sessions[st.session_state.session_id]["messages"]
    _cur_messages.append({
        "role": "user",
        "content": prompt,
    })
    # 首次对话时更新预览文本（以用户第一条消息作为预览）
    cur_sess = st.session_state.sessions[st.session_state.session_id]
    if cur_sess["preview"].startswith("对话 "):
        cur_sess["preview"] = prompt[:30]
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 运行 Agent
                total_start: float = time.perf_counter()
                result = graph_builder.invoke(
                    user_message=prompt,
                    session_id=st.session_state.session_id,
                )
                total_time: float = round(time.perf_counter() - total_start, 2)

                # 提取最终回答
                messages = result.get("messages", [])
                final_answer: str = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content
                        break

                if not final_answer:
                    final_answer = "抱歉，我无法处理您的请求，请稍后重试。"

                st.markdown(final_answer)

                # --- 收集 Tool 调用信息 ---
                tool_calls_info: List[Dict] = []
                retrieved_sources: List[Dict] = []

                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls_info.append({
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            })

                # --- 收集 Agent Monitor ---
                monitor_entries: List[Dict] = result.get("agent_monitor", [])
                for entry in monitor_entries:
                    entry["total_time"] = total_time

                # --- 收集 RAG 检索来源（增强版：含引擎信息）---
                tool_results: List[Dict] = result.get("tool_results", [])
                for tr in tool_results:
                    if tr.get("tool_name") == "insurance_rag_search":
                        result_text: str = tr.get("result", "")
                        # 根据内容判断引擎类型
                        engine_label: str = (
                            "LlamaIndex" if "引擎: LlamaIndex" in result_text
                            else "LangChain"
                        )
                        retrieved_sources.append({
                            "source": f"知识库检索结果 ({engine_label})",
                            "engine": engine_label,
                            "similarity": "N/A",
                            "content": result_text[:800],
                        })

                # 保存到会话状态
                _cur_messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tool_calls": tool_calls_info if tool_calls_info else None,
                    "retrieved_sources": retrieved_sources if retrieved_sources else None,
                    "agent_monitor": monitor_entries if monitor_entries else None,
                })

                # --- 展开式 Agent Monitor（增强版）---
                if monitor_entries:
                    with st.expander("📊 Agent Monitor（点击展开）"):
                        st.caption(f"⏱️ 总耗时: {total_time}s")
                        st.divider()

                        # Agent 决策步骤
                        agent_entries = [e for e in monitor_entries if e.get("step") == "agent"]
                        tool_entries = [e for e in monitor_entries if e.get("step") == "tool"]

                        for i, entry in enumerate(agent_entries, 1):
                            st.info(
                                f"**🧠 第{i}步: Agent 决策**\n\n"
                                f"意图识别: {entry.get('intent', 'N/A')}\n\n"
                                f"LLM 耗时: {entry.get('llm_time', 0):.2f}s"
                            )

                        for i, entry in enumerate(tool_entries, 1):
                            status: str = "✅ 成功" if entry.get("success") else "❌ 失败"
                            st.success(
                                f"**🔧 第{i}步: {entry.get('tool_name', 'N/A')}** {status}\n\n"
                                f"耗时: {entry.get('tool_time', 0):.4f}s"
                            )

                # --- 展开式 RAG 来源 ---
                if retrieved_sources:
                    with st.expander("📖 引用来源（点击展开）"):
                        for i, src in enumerate(retrieved_sources, 1):
                            st.markdown(f"**来源 {i}**")
                            st.caption(src.get("content", "")[:500])
                            st.divider()

            except Exception as e:
                st.error(f"处理请求时出错: {e}")
                logger.error(f"Agent 调用异常: {e}", exc_info=True)
