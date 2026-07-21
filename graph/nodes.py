"""
Insurance AI Agent - 节点实现模块
实现 LangGraph 中的 Agent 节点和 Tool 节点。
"""

import time
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from config import LLM_CONFIG, MEMORY_CONFIG
from graph.state import AgentState
from prompts.system_prompt import SYSTEM_PROMPT
from tools.insurance_rag_tool import InsuranceRAGTool
from tools.premium_calculator_tool import PremiumCalculatorTool
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("graph.nodes")

# ------------------------------------------------------------
# 全局工具实例（延迟初始化，支持热切换 RAG 引擎）
# ------------------------------------------------------------
_rag_tool: InsuranceRAGTool | None = None
_premium_tool: PremiumCalculatorTool | None = None
_tools_by_name: Dict[str, Any] = {}
_current_rag_engine: str = ""


def _get_tools() -> List[Any]:
    """
    获取所有已注册的工具实例。

    当 RAG_ENGINE 变更时自动重建工具，支持热切换。
    避免模块级缓存导致引擎切换不生效。

    Returns:
        工具列表
    """
    global _rag_tool, _premium_tool, _tools_by_name, _current_rag_engine

    from config import get_rag_engine
    engine = get_rag_engine()

    # 检测 RAG 引擎是否切换，切换时重建工具
    if _rag_tool is None or _current_rag_engine != engine:
        _rag_tool = InsuranceRAGTool()
        _current_rag_engine = engine

    if _premium_tool is None:
        _premium_tool = PremiumCalculatorTool()

    _tools_by_name = {
        _rag_tool.name: _rag_tool,
        _premium_tool.name: _premium_tool,
    }

    return [_rag_tool, _premium_tool]


def _get_llm() -> ChatOpenAI:
    """
    获取配置好的 LLM 实例（绑定工具）。

    Returns:
        ChatOpenAI 实例
    """
    tools: List[Any] = _get_tools()
    llm = ChatOpenAI(
        model=LLM_CONFIG["model"],
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
    )
    return llm.bind_tools(tools)


def _safe_truncate_messages(
    messages: List[Any], max_rounds: int
) -> List[Any]:
    """
    安全截断消息列表，确保不会在 AIMessage(tool_calls) 和后续
    ToolMessage 之间切断，避免 API 报错：
    "Messages with role 'tool' must be a response to a
     preceding message with 'tool_calls'"
    """
    if not messages:
        return []

    # 向前找到最后一个 HumanMessage 之前的"安全起点"
    # 确保每条 ToolMessage 都能找到对应的 tool_call_id
    human_count = 0
    start_idx = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, "type") and msg.type == "human":
            human_count += 1
            if human_count >= max_rounds:
                start_idx = i
                break
        elif isinstance(msg, HumanMessage):
            human_count += 1
            if human_count >= max_rounds:
                start_idx = i
                break
    else:
        start_idx = 0  # 不足 max_rounds 轮，全保留

    result = messages[start_idx:]

    # 修复起点：如果结果的第一条消息是 ToolMessage，
    # 向前追溯到对应的 AIMessage(tool_calls)
    while result and isinstance(result[0], ToolMessage):
        start_idx -= 1
        if start_idx < 0:
            # 放弃这条孤儿 ToolMessage
            result = result[1:]
        else:
            result = messages[start_idx:]

    # 修复终点遗留：去掉开头没有对应 tool_calls 的 ToolMessage
    # 逐个检查 ToolMessage 是否有前驱 AIMessage(tool_calls) 携带对应 id
    tool_call_ids: set = set()
    cleaned: List[Any] = []
    for msg in result:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_ids.add(tc.get("id", ""))
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "")
            if tc_id not in tool_call_ids:
                continue  # 孤儿 ToolMessage，丢弃
        cleaned.append(msg)

    return cleaned


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent 节点：调用 LLM，可能返回工具调用请求或最终回答。

    Args:
        state: Agent 当前状态

    Returns:
        包含 messages 和 agent_monitor 的字典
    """
    logger.info("[Node] Agent 节点开始执行")

    with Timer("LLM 调用") as timer:
        llm = _get_llm()

        # 构建消息列表：system prompt + 历史消息
        messages: List[Any] = state.get("messages", [])
        full_messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

        # 安全截断：确保不会在 tool_calls ↔ ToolMessage 之间切断
        max_rounds: int = MEMORY_CONFIG.get("max_context_rounds", 5)
        recent_messages: List[Any] = _safe_truncate_messages(messages, max_rounds)
        full_messages.extend(recent_messages)

        # 调用 LLM
        response: AIMessage = llm.invoke(full_messages)

    llm_time: float = round(timer.elapsed, 4)

    # 解析意图（从 tool_calls 推断）
    intent: str = "直接回答"
    if response.tool_calls:
        tool_names: List[str] = [
            tc.get("name", "") for tc in response.tool_calls
        ]
        intent = f"调用工具: {', '.join(tool_names)}"

    # 记录 Agent 执行信息
    monitor_entry: Dict[str, Any] = {
        "step": "agent",
        "intent": intent,
        "has_tool_calls": len(response.tool_calls) > 0,
        "tool_calls_count": len(response.tool_calls),
        "llm_time": llm_time,
        "timestamp": time.time(),
    }

    logger.info(
        f"[Node] Agent 节点完成: intent='{intent}', "
        f"{len(response.tool_calls)} 个 tool_calls, LLM耗时={llm_time}s"
    )

    return {
        "messages": [response],
        "intent": intent,
        "agent_monitor": [monitor_entry],
    }


def tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool 节点：执行 LLM 请求的工具调用，返回工具执行结果。

    Args:
        state: Agent 当前状态

    Returns:
        包含 messages、tool_results、retrieved_docs、agent_monitor 的字典
    """
    logger.info("[Node] Tools 节点开始执行")

    messages: List[Any] = state.get("messages", [])
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("[Node] Tools 节点：最后一条消息没有 tool_calls，跳过")
        return {"messages": []}

    tools: Dict[str, Any] = _tools_by_name
    tool_messages: List[ToolMessage] = []
    tool_results: List[Dict[str, Any]] = []
    retrieved_docs: List[Dict[str, Any]] = []
    monitor_entries: List[Dict[str, Any]] = []

    for tool_call in last_message.tool_calls:
        tool_name: str = tool_call.get("name", "unknown")
        tool_args: Dict[str, Any] = tool_call.get("args", {})
        tool_id: str = tool_call.get("id", "")

        logger.info(f"[Tool] 执行: {tool_name}({list(tool_args.keys())})")

        # 查找并执行工具
        if tool_name not in tools:
            result: str = f"错误：未知工具 '{tool_name}'"
            logger.error(result)
        else:
            tool_instance = tools[tool_name]
            start_time: float = time.perf_counter()

            try:
                result = tool_instance._run(**tool_args)
            except Exception as e:
                result = f"工具执行失败: {e}"
                logger.error(f"[Tool] {tool_name} 执行异常: {e}", exc_info=True)

            elapsed: float = round(time.perf_counter() - start_time, 4)

            # 如果是 RAG 工具，保存检索到的文档
            if tool_name == "insurance_rag_search" and hasattr(
                tool_instance, "_retriever"
            ):
                # 提取检索结果中添加来源信息
                retrieved_docs.append({
                    "tool_name": tool_name,
                    "query": tool_args.get("query", ""),
                    "elapsed": elapsed,
                })

            monitor_entries.append({
                "step": "tool",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": str(result)[:500],
                "tool_time": elapsed,
                "success": not result.startswith("错误"),
                "timestamp": time.time(),
            })

            tool_results.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": str(result)[:500],
                "elapsed": elapsed,
            })

        # 包装为 ToolMessage
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
        )

    logger.info(
        f"[Node] Tools 节点完成: 执行 {len(last_message.tool_calls)} 个工具"
    )

    return {
        "messages": tool_messages,
        "tool_results": tool_results,
        "retrieved_docs": retrieved_docs,
        "agent_monitor": monitor_entries,
    }
