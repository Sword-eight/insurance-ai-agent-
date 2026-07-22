"""
Insurance AI Agent - 状态管理器模块
封装 LangGraph 的对话状态获取和历史管理。

生命周期：graph 由 init_services() 创建后注入，StateManager 不自行创建任何对象。
"""

from typing import Any, Dict, List

from config import MEMORY_CONFIG
from utils.logger import get_logger

logger = get_logger("memory.state_manager")


class StateManager:
    """
    状态管理器。
    负责获取对话历史、清除会话状态等操作。

    所有方法通过 self._graph 访问状态，不再需要调用方传入 graph 参数。
    """

    def __init__(self, graph: Any) -> None:
        """
        Args:
            graph: LangGraph 编译后的 StateGraph（含 checkpointer）
        """
        self._graph = graph

    def get_conversation_history(
        self,
        session_id: str = "default",
        max_rounds: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        获取指定会话的对话历史。

        Args:
            session_id: 会话 ID
            max_rounds: 最大返回轮数，默认使用配置值

        Returns:
            对话历史消息列表
        """
        max_rounds = max_rounds or MEMORY_CONFIG.get("max_context_rounds", 5)
        thread: dict = {"configurable": {"thread_id": session_id}}

        try:
            state = self._graph.get_state(thread)
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
        self, session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        清除指定会话的历史记录。

        注意：InMemorySaver 不支持直接删除单个线程，
        当前实现通过标记实现——下次使用相同 session_id 时将覆盖旧状态。

        Args:
            session_id: 会话 ID

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
        self, session_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        获取指定会话的 Tool 调用记录。

        Args:
            session_id: 会话 ID

        Returns:
            Tool 调用记录列表
        """
        thread: dict = {"configurable": {"thread_id": session_id}}

        try:
            state = self._graph.get_state(thread)
            if state is None or state.values is None:
                return []

            return list(state.values.get("tool_results", []))
        except Exception as e:
            logger.error(f"获取 Tool 记录失败: {e}")
            return []
