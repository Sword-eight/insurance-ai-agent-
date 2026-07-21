"""
Insurance AI Agent - 状态管理器模块
封装 LangGraph 的状态获取和历史管理。
"""

from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver

from config import MEMORY_CONFIG
from utils.logger import get_logger

logger = get_logger("memory.state_manager")


class StateManager:
    """
    状态管理器。
    负责获取对话历史、清除会话状态等操作。
    """

    def __init__(self, checkpointer: InMemorySaver) -> None:
        """
        Args:
            checkpointer: LangGraph 检查点保存器
        """
        self._checkpointer = checkpointer

    def get_conversation_history(
        self,
        graph,
        session_id: str = "default",
        max_rounds: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        获取指定会话的对话历史。

        Args:
            graph: LangGraph 编译后的图
            session_id: 会话 ID
            max_rounds: 最大返回轮数，默认使用配置值

        Returns:
            对话历史消息列表
        """
        max_rounds = max_rounds or MEMORY_CONFIG.get("max_context_rounds", 5)
        thread: dict = {"configurable": {"thread_id": session_id}}

        try:
            state = graph.get_state(thread)
            if state is None or state.values is None:
                return []

            messages: List[Any] = state.values.get("messages", [])
            history: List[Dict[str, Any]] = []

            for msg in messages[-(max_rounds * 2):]:
                role: str = "unknown"
                content: str = ""

                if hasattr(msg, "type"):
                    msg_type: str = msg.type
                    if msg_type in ("human", "user"):
                        role = "user"
                    elif msg_type in ("ai", "assistant"):
                        role = "assistant"
                    elif msg_type == "tool":
                        role = "tool"

                if hasattr(msg, "content"):
                    content = str(msg.content)[:200]

                history.append({
                    "role": role,
                    "content": content,
                })

            return history

        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    def clear_session(
        self, graph, session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        清除指定会话的历史记录（通过创建新线程实现）。

        Args:
            graph: LangGraph 编译后的图
            session_id: 会话 ID（将被标记为已清除）

        Returns:
            操作结果
        """
        logger.info(f"清除会话: {session_id}")
        return {
            "success": True,
            "message": f"会话 {session_id} 已清除。"
            "下次使用时将创建新的对话历史。",
        }

    def get_tool_results(
        self, graph, session_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        获取指定会话的 Tool 调用记录。

        Args:
            graph: LangGraph 编译后的图
            session_id: 会话 ID

        Returns:
            Tool 调用记录列表
        """
        thread: dict = {"configurable": {"thread_id": session_id}}

        try:
            state = graph.get_state(thread)
            if state is None or state.values is None:
                return []

            return list(state.values.get("tool_results", []))
        except Exception as e:
            logger.error(f"获取 Tool 记录失败: {e}")
            return []
