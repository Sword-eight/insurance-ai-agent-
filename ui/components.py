"""
可复用 UI 组件模块。
跨页面共享的 Streamlit 渲染片段。

设计原则：
  - 只做渲染（st.xxx() 调用），不包含业务逻辑
  - 接收数据 + 可选回调，不访问 st.session_state
"""

from typing import Any, Dict, List

import streamlit as st


# ==================================================================
# 知识库状态卡片
# ==================================================================

def kb_stats_cards(stats: Dict[str, Any]) -> None:
    """渲染知识库统计指标卡片（PDF 数量 / 向量数量）。"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDF 数量", stats.get("pdf_count", 0))
    with col2:
        st.metric("向量数量", stats.get("total_vectors", 0))


def kb_stats_detail(stats: Dict[str, Any], llm_model: str) -> None:
    """渲染知识库统计详情（在 expander 内）。"""
    st.markdown(f"""
    - **Embedding 模型**: {stats.get('embedding_model', 'N/A')}
    - **Chunk Size**: {stats.get('chunk_size', 'N/A')}
    - **Chunk Overlap**: {stats.get('chunk_overlap', 'N/A')}
    - **索引状态**: {'已加载' if stats.get('index_loaded') else '未加载'}
    - **LLM 模型**: {llm_model}
    """)


# ==================================================================
# 线程列表
# ==================================================================

def thread_status_icon(is_active: bool) -> str:
    """返回线程状态图标。"""
    return "🟢 " if is_active else "⚪ "


def thread_list_item(
    session_id: str,
    session_data: Dict[str, Any],
    is_active: bool,
) -> None:
    """渲染单个线程的列表项。"""
    prefix = thread_status_icon(is_active)
    num = session_data.get("_number", "?")
    preview = session_data.get("preview", "")[:16]
    st.text(f"{prefix}#{num} {preview}")


# ==================================================================
# 消息渲染
# ==================================================================

def tool_calls_expander(tool_calls: List[Dict]) -> None:
    """渲染工具调用详情 expander。"""
    with st.expander("🔧 查看工具调用"):
        for tc in tool_calls:
            st.json(tc)


def rag_sources_expander(sources: List[Dict]) -> None:
    """渲染 RAG 检索来源 expander。"""
    with st.expander("📖 查看引用来源"):
        for i, source in enumerate(sources, 1):
            engine = source.get("engine", "LangChain")
            st.markdown(f"**来源 {i}** ({engine})")
            st.caption(source.get("content", "")[:500])
            st.divider()


def agent_monitor_agent_step(entry: Dict[str, Any]) -> None:
    """渲染 Agent 决策步骤。"""
    st.info(
        f"**🧠 Agent 决策**\n\n"
        f"意图识别: {entry.get('intent', 'N/A')}\n\n"
        f"LLM 耗时: {entry.get('llm_time', 0):.2f}s"
    )


def agent_monitor_tool_step(entry: Dict[str, Any]) -> None:
    """渲染工具执行步骤。"""
    status = "✅ 成功" if entry.get("success") else "❌ 失败"
    st.success(
        f"**🔧 {entry.get('tool_name', 'N/A')}** {status}\n\n"
        f"耗时: {entry.get('tool_time', 0):.4f}s"
    )


def agent_monitor_expander(
    monitor_entries: List[Dict],
    total_time: float,
) -> None:
    """渲染完整的 Agent Monitor expander。"""
    with st.expander("📊 Agent Monitor（点击展开）"):
        st.caption(f"⏱️ 总耗时: {total_time}s")
        st.divider()

        agent_entries = [e for e in monitor_entries if e.get("step") == "agent"]
        tool_entries = [e for e in monitor_entries if e.get("step") == "tool"]

        for i, entry in enumerate(agent_entries, 1):
            agent_monitor_agent_step(entry)

        for i, entry in enumerate(tool_entries, 1):
            agent_monitor_tool_step(entry)


# ==================================================================
# 开发者面板
# ==================================================================

def developer_panel(
    current_engine: str,
    kb_stats: Dict[str, Any],
    retriever_type: str,
    top_k: int,
) -> None:
    """渲染开发者调试面板。"""
    st.markdown(f"""
    - **Current RAG Engine**: {current_engine}
    - **Embedding Model**: {kb_stats.get('embedding_model', 'N/A')}
    - **Vector Store**: FAISS
    - **Retriever Type**: {retriever_type}
    - **Chunk Size**: {kb_stats.get('chunk_size', 'N/A')}
    - **Chunk Overlap**: {kb_stats.get('chunk_overlap', 'N/A')}
    - **Top-K**: {top_k}
    - **PDF Count**: {kb_stats.get('pdf_count', 0)}
    - **Total Vectors**: {kb_stats.get('total_vectors', 0)}
    """)
