"""
Insurance AI Agent - 路由模块
根据 LLM 的回复决定下一步：调用工具 or 结束。
"""

from typing import Literal

from langchain_core.messages import AIMessage

from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger("graph.router")


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """
    检查 Agent 节点的输出，决定路由目标。

    路由规则：
    - 如果最后一条消息包含 tool_calls → 进入 tools 节点
    - 否则 → 结束对话

    Args:
        state: Agent 当前状态

    Returns:
        "tools" 或 "__end__"
    """
    messages = state.get("messages", [])
    if not messages:
        logger.warning("Router: 消息列表为空，结束对话")
        return "__end__"

    last_message = messages[-1]

    # 检查是否有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_names: list[str] = [
            tc.get("name", "unknown") for tc in last_message.tool_calls
        ]
        logger.info(f"Router: 检测到 {len(tool_names)} 个工具调用 → tools 节点 ({tool_names})")
        return "tools"

    logger.info("Router: 无工具调用 → 结束对话")
    return "__end__"
