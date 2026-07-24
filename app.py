"""
Insurance AI Agent - 主入口
基于 LangGraph + Streamlit 的保险智能助手 Web 界面。

职责：纯组装入口。初始化服务 → 渲染 UI → 绑定事件。
不包含任何业务逻辑、不直接操作 session state。

启动方式：
    streamlit run app.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import STREAMLIT_CONFIG, get_rag_engine

# ── 页面配置 ────────────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
)

# ── 初始化服务 + 会话 ──────────────────────────────────────────
from application.bootstrap import init_services
from application.session import init as init_sessions
from application.session import get_all, get_active_id, get_messages, create

if "_current_rag_engine" not in st.session_state:
    st.session_state._current_rag_engine = get_rag_engine()

with st.spinner("正在初始化系统..."):
    services = init_services(rag_engine=st.session_state._current_rag_engine)

init_sessions()

knowledge_service = services["knowledge_service"]
graph_builder = services["graph_builder"]
retrieval_service = services["retrieval_service"]

# ── 侧边栏 ──────────────────────────────────────────────────────
from application.handlers import (
    handle_pdf_upload,
    handle_rebuild_index,
    handle_delete_index,
)
from ui.sidebar import render as render_sidebar


def _on_engine_change(new_engine: str) -> None:
    """引擎切换回调：更新状态 → 清缓存 → 重跑。"""
    st.session_state._current_rag_engine = new_engine
    st.cache_resource.clear()
    st.rerun()


render_sidebar(
    knowledge_service=knowledge_service,
    retrieval_service=retrieval_service,
    current_engine=st.session_state._current_rag_engine,
    on_engine_change=_on_engine_change,
    on_upload=lambda files: handle_pdf_upload(files, knowledge_service),
    on_rebuild=lambda: handle_rebuild_index(knowledge_service),
    on_delete=lambda: handle_delete_index(knowledge_service),
    project_root=PROJECT_ROOT,
)

# ── 主聊天区 ────────────────────────────────────────────────────
from application.handlers import handle_chat_message
from ui.chat import render as render_chat


def _on_send(prompt: str):
    """发送消息回调。"""
    return handle_chat_message(
        prompt=prompt,
        graph_builder=graph_builder,
        session_id=get_active_id(),
    )


render_chat(
    messages=get_messages(),
    sessions=get_all(),
    active_id=get_active_id(),
    on_create_thread=create,
    on_send=_on_send,
)
