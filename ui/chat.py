"""
聊天窗口 UI 模块。
渲染主聊天区域：线程切换器、消息历史、聊天输入、Agent 响应。

设计原则：
  - 只做渲染（st.xxx()），不包含业务逻辑
  - 通过回调函数与事件层解耦
"""

from typing import Any, Callable, Dict, List

import streamlit as st

from config import STREAMLIT_CONFIG
from ui.components import (
    tool_calls_expander,
    rag_sources_expander,
    agent_monitor_expander,
)


# ==================================================================
# 线程切换器
# ==================================================================

def render_thread_switcher(
    sessions: Dict[str, Dict[str, Any]],
    active_id: str,
    on_create: Callable[[], str],
) -> None:
    """
    渲染线程切换下拉框 + 新建按钮。

    Args:
        sessions:   全部会话字典
        active_id:  当前活跃会话 ID
        on_create:  新建线程回调（返回新线程 ID）
    """
    sids = list(sessions.keys())

    # 分配稳定序号
    for i, tid in enumerate(sids, 1):
        sessions[tid]["_number"] = i

    labels = [
        f"#{sessions[t].get('_number', '?')} — {sessions[t]['preview'][:16]}"
        for t in sids
    ]
    cur = sids.index(active_id) if active_id in sids else 0

    picked = st.selectbox(
        "🧵 切换对话线程",
        options=range(len(sids)),
        format_func=lambda i: labels[i],
        index=cur,
        key="thread_selector_v2",
    )

    # 切换线程
    if sids[picked] != active_id:
        from application.session import switch_to
        switch_to(sids[picked])

    if st.button("➕ 新建对话线程", key="new_thread_btn"):
        new_id = on_create()
        st.rerun()


# ==================================================================
# 消息历史
# ==================================================================

def render_message_history(messages: List[Dict[str, Any]]) -> None:
    """
    渲染当前线程的全部历史消息。

    Args:
        messages: 消息列表
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("tool_calls"):
                tool_calls_expander(msg["tool_calls"])

            if msg.get("retrieved_sources"):
                rag_sources_expander(msg["retrieved_sources"])

            if msg.get("agent_monitor"):
                agent_monitor_expander(
                    msg["agent_monitor"],
                    total_time=0.0,  # 历史消息无实时耗时
                )


# ==================================================================
# Agent 响应渲染
# ==================================================================

def render_agent_response(result: Dict[str, Any]) -> None:
    """
    渲染 Agent 的完整响应（最终回答 + 可选 expander）。

    Args:
        result: handle_chat_message() 的返回值
    """
    final_answer = result["final_answer"]
    tool_calls = result.get("tool_calls") or []
    sources = result.get("retrieved_sources") or []
    monitor = result.get("agent_monitor") or []
    total_time = result.get("total_time", 0.0)

    st.markdown(final_answer)

    if monitor:
        agent_monitor_expander(monitor, total_time)

    if sources:
        with st.expander("📖 引用来源（点击展开）"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**来源 {i}**")
                st.caption(src.get("content", "")[:500])
                st.divider()


# ==================================================================
# 聊天主入口
# ==================================================================

def render(
    *,
    messages: List[Dict[str, Any]],
    sessions: Dict[str, Dict[str, Any]],
    active_id: str,
    on_create_thread: Callable[[], str],
    on_send: Callable[[str], Dict[str, Any]],
) -> None:
    """
    渲染完整的聊天界面。

    Args:
        messages:         当前线程消息列表
        sessions:         全部会话字典
        active_id:        当前活跃会话 ID
        on_create_thread: 新建线程回调
        on_send:          发送消息回调（接收 prompt，返回解析后的结果）
    """
    st.title("🛡️ Insurance AI Agent")
    st.caption("保险智能助手 — 您的专业保险顾问")

    # 线程切换器
    render_thread_switcher(sessions, active_id, on_create_thread)
    st.divider()

    # 消息历史
    render_message_history(messages)

    # 聊天输入
    if prompt := st.chat_input("请输入您的保险相关问题..."):
        _handle_user_input(prompt, on_send)


def _handle_user_input(
    prompt: str,
    on_send: Callable[[str], Dict[str, Any]],
) -> None:
    """
    处理用户输入：保存用户消息 → 调用 Agent → 渲染响应 → 保存结果。

    Args:
        prompt:  用户输入文本
        on_send: 发送消息回调
    """
    from application.session import get_active_id, add_message, update_preview

    # 渲染用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = on_send(prompt)
                render_agent_response(result)

                # 保存到会话
                session_id = get_active_id()
                add_message("user", prompt, session_id=session_id)
                update_preview(prompt, session_id=session_id)
                add_message(
                    "assistant",
                    result["final_answer"],
                    tool_calls=result.get("tool_calls") or None,
                    retrieved_sources=result.get("retrieved_sources") or None,
                    agent_monitor=result.get("agent_monitor") or None,
                    session_id=session_id,
                )

            except Exception as e:
                st.error(f"处理请求时出错: {e}")
                from utils.logger import get_logger
                get_logger("ui.chat").error(f"Agent 调用异常: {e}", exc_info=True)
