"""
多会话状态管理模块。
封装 st.session_state 中 sessions 字典的全部增删改查操作。

所有 session 操作通过此模块进行，不直接在 app.py 中操作 st.session_state。
"""

import time
from typing import Any, Dict, List

import streamlit as st

# Session 数据中每条消息的类型
MessageDict = Dict[str, Any]
SessionDict = Dict[str, Any]

# ── 默认 session key 常量 ───────────────────────────────────────
_SESSIONS_KEY = "sessions"
_SESSION_ID_KEY = "session_id"


# ==================================================================
# 初始化
# ==================================================================

def init() -> None:
    """初始化多会话状态（仅在首次运行时执行）。"""
    if _SESSIONS_KEY not in st.session_state:
        default_id = f"session_{int(time.time())}"
        st.session_state[_SESSIONS_KEY] = {
            default_id: {
                "id": default_id,
                "messages": [],
                "created_at": time.strftime("%H:%M:%S"),
                "preview": "对话 1",
                "_number": 1,
            }
        }
        st.session_state[_SESSION_ID_KEY] = default_id


# ==================================================================
# 查询
# ==================================================================

def get_all() -> Dict[str, SessionDict]:
    """获取全部会话。"""
    return st.session_state.get(_SESSIONS_KEY, {})


def get_active_id() -> str:
    """获取当前活跃会话 ID。"""
    return st.session_state.get(_SESSION_ID_KEY, "")


def get_active() -> SessionDict | None:
    """获取当前活跃会话。"""
    sid = get_active_id()
    return get_all().get(sid)


def get_messages(session_id: str | None = None) -> List[MessageDict]:
    """获取指定会话的消息列表。"""
    sid = session_id or get_active_id()
    sess = get_all().get(sid)
    return sess["messages"] if sess else []


def count() -> int:
    """获取会话总数。"""
    return len(get_all())


def get_ids() -> List[str]:
    """获取按创建顺序排列的会话 ID 列表。"""
    return list(get_all().keys())


# ==================================================================
# 修改
# ==================================================================

def create() -> str:
    """创建新会话并设为活跃，返回新会话 ID。"""
    nid = f"sess_{int(time.time())}"
    next_num = count() + 1
    st.session_state[_SESSIONS_KEY][nid] = {
        "id": nid,
        "messages": [],
        "created_at": time.strftime("%H:%M:%S"),
        "preview": f"对话 {next_num}",
        "_number": next_num,
    }
    st.session_state[_SESSION_ID_KEY] = nid
    return nid


def switch_to(session_id: str) -> None:
    """切换到指定会话。"""
    if session_id in st.session_state.get(_SESSIONS_KEY, {}):
        st.session_state[_SESSION_ID_KEY] = session_id


def add_message(
    role: str,
    content: str,
    tool_calls: List[Dict] | None = None,
    retrieved_sources: List[Dict] | None = None,
    agent_monitor: List[Dict] | None = None,
    session_id: str | None = None,
) -> None:
    """向指定会话添加一条消息。"""
    sid = session_id or get_active_id()
    sessions = get_all()
    if sid not in sessions:
        return

    sessions[sid]["messages"].append({
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "retrieved_sources": retrieved_sources,
        "agent_monitor": agent_monitor,
    })


def update_preview(text: str, session_id: str | None = None) -> None:
    """更新会话预览文本（用于侧边栏显示）。"""
    sid = session_id or get_active_id()
    sess = get_all().get(sid)
    if sess and sess["preview"].startswith("对话 "):
        sess["preview"] = text[:30]
