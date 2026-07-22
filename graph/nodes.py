"""
Insurance AI Agent - 节点辅助函数
提供消息安全截断等纯工具函数。
Agent/Tool 节点实现已移入 AgentGraphBuilder（通过依赖注入访问 tools）。
"""

from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def safe_truncate_messages(
    messages: List[Any], max_rounds: int
) -> List[Any]:
    """
    安全截断消息列表，确保不会在 AIMessage(tool_calls) 和后续
    ToolMessage 之间切断，避免 API 报错：
    "Messages with role 'tool' must be a response to a
     preceding message with 'tool_calls'"

    这是一个纯函数，不依赖任何全局状态。

    Args:
        messages: 完整的消息列表
        max_rounds: 保留的最大对话轮数

    Returns:
        安全截断后的消息列表
    """
    if not messages:
        return []

    # 向前找到最后一个 HumanMessage 之前的"安全起点"
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
            result = result[1:]
        else:
            result = messages[start_idx:]

    # 修复：去掉开头没有对应 tool_calls 的 ToolMessage
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
