"""
Insurance AI Agent - LangGraph 图构建模块
使用 LangGraph 构建 Agent 工作流：Agent ↔ Tool 循环。
"""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import agent_node, tools_node
from graph.router import route_after_agent
from utils.logger import get_logger

logger = get_logger("graph.builder")


class AgentGraphBuilder:
    """
    LangGraph Agent 图构建器。
    构建 Agent ↔ Tool 的循环工作流，支持多轮工具调用。
    """

    def __init__(self) -> None:
        """初始化图构建器。"""
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph()
        logger.info("LangGraph Agent 图构建完成")

    def _build_graph(self):
        """
        构建 Agent 工作流图。

        流程：
            START → agent → [router] → tools → agent → END
                           └──────────────→ END

        Returns:
            编译后的 StateGraph
        """
        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)

        # 设置入口
        builder.set_entry_point("agent")

        # 添加条件边：agent 之后 → tools 或 END
        builder.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tools",
                "__end__": END,
            },
        )

        # tools 执行完回到 agent 继续思考
        builder.add_edge("tools", "agent")

        # 编译图（带记忆）
        return builder.compile(checkpointer=self._checkpointer)

    @property
    def graph(self):
        """获取编译后的 LangGraph 图。"""
        return self._graph

    @property
    def checkpointer(self) -> InMemorySaver:
        """获取内存检查点保存器。"""
        return self._checkpointer

    def invoke(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> dict:
        """
        运行 Agent 处理用户消息（同步方式）。

        Args:
            user_message: 用户输入文本
            session_id: 会话 ID（用于多轮对话记忆）

        Returns:
            Agent 最终状态
        """
        from langchain_core.messages import HumanMessage

        thread: dict = {"configurable": {"thread_id": session_id}}

        initial_state: dict = {
            "messages": [HumanMessage(content=user_message)],
            "tool_results": [],
            "retrieved_docs": [],
            "session_id": session_id,
            "intent": "",
            "agent_monitor": [],
        }

        logger.info(f"[Graph] 开始执行: session={session_id}, msg='{user_message[:50]}...'")
        result = self._graph.invoke(initial_state, thread)
        logger.info(f"[Graph] 执行完成: session={session_id}")

        return result

    def stream(
        self,
        user_message: str,
        session_id: str = "default",
    ):
        """
        流式运行 Agent（逐步返回每个节点的输出）。

        Args:
            user_message: 用户输入文本
            session_id: 会话 ID

        Yields:
            每个节点的事件
        """
        from langchain_core.messages import HumanMessage

        thread: dict = {"configurable": {"thread_id": session_id}}

        initial_state: dict = {
            "messages": [HumanMessage(content=user_message)],
            "tool_results": [],
            "retrieved_docs": [],
            "session_id": session_id,
            "intent": "",
            "agent_monitor": [],
        }

        logger.info(f"[Graph] 开始流式执行: session={session_id}")
        for event in self._graph.stream(initial_state, thread):
            yield event
        logger.info(f"[Graph] 流式执行完成: session={session_id}")

    def get_state(self, session_id: str = "default") -> Any:
        """
        获取指定会话的当前状态。

        Args:
            session_id: 会话 ID

        Returns:
            会话状态快照
        """
        thread: dict = {"configurable": {"thread_id": session_id}}
        return self._graph.get_state(thread)
