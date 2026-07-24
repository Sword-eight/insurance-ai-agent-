"""
侧边栏 UI 模块。
渲染全部侧边栏内容：线程列表、知识库状态、管理员面板、日志查看器、开发者面板。

设计原则：
  - 只做渲染（st.xxx()），不包含业务逻辑
  - 通过回调函数与事件层解耦
  - 通过 st.session_state 读取运行时状态（引擎切换等）
"""

from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st

from config import LLM_CONFIG, STREAMLIT_CONFIG, VECTOR_STORE_CONFIG, PDF_CONFIG
from ui.components import (
    kb_stats_cards,
    kb_stats_detail,
    thread_list_item,
    developer_panel,
)


def render(
    *,
    knowledge_service: Any,
    retrieval_service: Any,
    current_engine: str,
    on_engine_change: Callable[[str], None],
    on_upload: Callable[[List[Any]], Dict[str, Any]],
    on_rebuild: Callable[[], Dict[str, Any]],
    on_delete: Callable[[], Dict[str, Any]],
    project_root: Path,
) -> None:
    """
    渲染侧边栏全部内容。

    Args:
        knowledge_service: KnowledgeService 实例（索引管理）
        retrieval_service: RetrievalService 实例（检索入口）
        current_engine:   当前 RAG 引擎名
        on_engine_change: 引擎切换回调（接收新引擎名）
        on_upload:        PDF 上传回调（接收文件列表，返回结果）
        on_rebuild:       重建索引回调（返回结果）
        on_delete:        删除索引回调（返回结果）
        project_root:     项目根目录（用于日志路径）
    """
    with st.sidebar:
        st.title("🛡️ Insurance AI Agent")
        st.divider()

        _render_thread_section()
        st.divider()
        _render_kb_status(knowledge_service)
        st.divider()
        _render_engine_selector(current_engine, on_engine_change)
        st.divider()
        _render_admin_panel(knowledge_service, on_upload, on_rebuild, on_delete)
        st.divider()
        _render_pdf_file_list(knowledge_service)
        st.divider()
        _render_log_viewer(project_root)
        st.divider()
        _render_developer_panel(knowledge_service, retrieval_service, current_engine)
        _render_about()


# ==================================================================
# 各区块渲染
# ==================================================================

def _render_thread_section() -> None:
    """渲染线程管理区块。"""
    st.subheader("🧵 线程管理")

    from application.session import get_all, get_active_id

    sessions = get_all()
    active_id = get_active_id()

    st.caption(f"共 {len(sessions)} 个对话线程，在顶部下拉框切换")
    for tid, sess in sessions.items():
        thread_list_item(
            session_id=tid,
            session_data=sess,
            is_active=(tid == active_id),
        )


def _render_kb_status(knowledge_service: Any) -> None:
    """渲染知识库状态区块。"""
    st.subheader("📚 Knowledge Base")
    stats = knowledge_service.get_stats()

    kb_stats_cards(stats)

    with st.expander("查看详情"):
        kb_stats_detail(stats, LLM_CONFIG["model"])


def _render_engine_selector(
    current_engine: str,
    on_change: Callable[[str], None],
) -> None:
    """渲染 RAG 引擎选择器。"""
    st.subheader("🔧 RAG 引擎")
    options = ["langchain", "llamaindex"]
    idx = 0 if current_engine == "langchain" else 1

    selected = st.selectbox(
        "选择 RAG 引擎",
        options=options,
        index=idx,
        help="LangChain: PyMuPDF + RecursiveCharacterTextSplitter + FAISS\n"
             "LlamaIndex: SimpleDirectoryReader + SentenceSplitter + VectorStoreIndex + FAISS",
    )

    if selected != current_engine:
        on_change(selected)


def _render_admin_panel(
    knowledge_service: Any,
    on_upload: Callable,
    on_rebuild: Callable,
    on_delete: Callable,
) -> None:
    """渲染管理员功能区块（PDF 上传 + 索引管理）。"""
    st.subheader("⚙️ 管理员")

    # PDF 上传
    uploaded_files = st.file_uploader(
        "上传 PDF 文档",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传保险条款 PDF 文档到知识库",
    )

    if uploaded_files:
        if st.button("📤 上传并处理", use_container_width=True):
            with st.spinner("正在重建知识库索引..."):
                result = on_upload(uploaded_files)
            if result["success"]:
                st.success(f"已上传 {result['saved_count']} 个 PDF，{result['message']}")
                st.rerun()
            else:
                st.error(result["message"])

    # 管理按钮
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 重建索引", use_container_width=True):
            with st.spinner("正在重建知识库..."):
                result = on_rebuild()
            if result["success"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])

    with col_b:
        if st.button("🗑️ 删除知识库", use_container_width=True):
            if st.warning("确定删除知识库索引？"):
                result = on_delete()
                st.success(result["message"])
                st.rerun()


def _render_pdf_file_list(knowledge_service: Any) -> None:
    """渲染 PDF 文件列表。"""
    st.subheader("📄 PDF 文件列表")
    pdf_files = knowledge_service.get_pdf_files()
    if pdf_files:
        for f in pdf_files:
            st.text(f"📎 {f['name']} ({f['size']:,} bytes)")
    else:
        st.caption("暂无 PDF 文件")


def _render_log_viewer(project_root: Path) -> None:
    """渲染日志查看器。"""
    st.subheader("📋 日志查看")
    with st.expander("查看最近日志"):
        log_dir = project_root / "logs"
        log_files = sorted(
            log_dir.glob("agent_*.log"), reverse=True
        ) if log_dir.exists() else []

        if log_files:
            selected = st.selectbox(
                "选择日志文件",
                [f.name for f in log_files[:7]],
                label_visibility="collapsed",
            )
            if selected:
                log_path = log_dir / selected
                try:
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()[-50:]
                    st.code("".join(reversed(lines)), language="log")
                except Exception:
                    st.caption("无法读取日志文件")
        else:
            st.caption("暂无日志文件")


def _render_developer_panel(
    knowledge_service: Any,
    retrieval_service: Any,
    current_engine: str,
) -> None:
    """渲染开发者调试面板。"""
    with st.expander("🛠️ Developer Panel"):
        stats = knowledge_service.get_stats()
        developer_panel(
            current_engine=current_engine,
            kb_stats=stats,
            retriever_type=retrieval_service.retriever_type,
            top_k=retrieval_service.top_k,
        )


def _render_about() -> None:
    """渲染关于信息。"""
    with st.expander("ℹ️ 关于"):
        st.markdown("""
        **Insurance AI Agent**

        基于 LangGraph 构建的企业级保险智能助手。

        - Agent + Tool Calling + RAG
        - DeepSeek LLM
        - BAAI/bge-base-zh-v1.5 Embedding
        - FAISS 向量数据库
        """)
