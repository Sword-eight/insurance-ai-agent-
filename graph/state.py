"""
Insurance AI Agent - Agent 状态定义模块
定义 LangGraph 使用的 AgentState，包含对话历史、工具调用记录等。
"""

from typing import Annotated, Any, Dict, List, TypedDict
import operator

from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    """
    Agent 状态。
    LangGraph 中所有节点共享此状态。
    使用 Annotated + operator.add 实现消息的自动追加。
    """

    # 对话消息历史（使用 operator.add reducer 自动累加）
    messages: Annotated[List[AnyMessage], operator.add]

    # 工具调用日志（用于 Agent Monitor 展示）
    tool_results: Annotated[List[Dict[str, Any]], operator.add]

    # RAG 检索到的文档（来源追溯）
    retrieved_docs: Annotated[List[Dict[str, Any]], operator.add]

    # 会话 ID
    session_id: str

    # Agent 意图识别结果（用于可视化）
    intent: str

    # Agent 执行过程记录（Agent Monitor 数据）
    agent_monitor: Annotated[List[Dict[str, Any]], operator.add]
