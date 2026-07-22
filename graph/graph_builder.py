"""
Insurance AI Agent - LangGraph 图构建模块
使用 LangGraph 构建 Agent 工作流：Agent ↔ Tool 循环。

生命周期：
  - 所有依赖通过构造函数注入（llm / checkpointer / system_prompt 等）
  - agent_node / tools_node 作为实例方法，通过 self 访问全部依赖
  - 零个模块级全局变量，零个隐式依赖
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from config import LLM_CONFIG, MEMORY_CONFIG
from graph.state import AgentState
from graph.router import route_after_agent
from graph.nodes import safe_truncate_messages
from prompts.system_prompt import SYSTEM_PROMPT
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("graph.builder")


# ==================================================================
# 默认工厂（仅在调用方不注入时使用）
# ==================================================================

def _create_default_llm() -> ChatOpenAI:
    """创建默认 LLM 实例（从 config.py 读取配置）。"""
    return ChatOpenAI(
        model=LLM_CONFIG["model"],
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
    )


# ==================================================================
# AgentGraphBuilder
# ==================================================================

class AgentGraphBuilder:
    """
    LangGraph Agent 图构建器。
    构建 Agent ↔ Tool 的循环工作流，支持多轮工具调用。

    全部依赖通过构造函数注入（有合理默认值）：
      - tools:               Tool 实例列表（必填）
      - llm:                 预配置的 ChatOpenAI 实例
      - checkpointer:        LangGraph 检查点保存器
      - system_prompt:       Agent 系统提示词
      - max_context_rounds:  最大对话上下文轮数

    使用方式：
        tools = [InsuranceRAGTool(retriever=...), PremiumCalculatorTool(service=...)]
        graph_builder = AgentGraphBuilder(tools=tools)
        result = graph_builder.invoke(user_message="等待期多久？", session_id="...")
    """

    def __init__(
        self,
        tools: List[Any],
        *,
        llm: Optional[ChatOpenAI] = None,
        checkpointer: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        max_context_rounds: Optional[int] = None,
    ) -> None:
        """
        Args:
            tools:               LangChain BaseTool 实例列表（必填）
            llm:                  ChatOpenAI 实例（默认从 LLM_CONFIG 创建）
            checkpointer:        LangGraph checkpointer（默认 InMemorySaver）
            system_prompt:       Agent 系统提示词（默认从 prompts 模块读取）
            max_context_rounds:  最大上下文轮数（默认从 MEMORY_CONFIG 读取）
        """
        # 依赖注入（全部有默认值，实现向后兼容）
        self._tools = tools
        self._tools_by_name: Dict[str, Any] = {t.name: t for t in tools}
        self._llm = llm or _create_default_llm()
        self._checkpointer = checkpointer or InMemorySaver()
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._max_context_rounds = (
            max_context_rounds
            if max_context_rounds is not None
            else MEMORY_CONFIG.get("max_context_rounds", 5)
        )

        self._graph = self._build_graph()

        logger.info(
            f"LangGraph Agent 图构建完成: "
            f"tools={list(self._tools_by_name.keys())}, "
            f"llm={self._llm.model_name}, "
            f"checkpointer={type(self._checkpointer).__name__}, "
            f"max_rounds={self._max_context_rounds}"
        )

    # ------------------------------------------------------------------
    # 节点实现
    # ------------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent 节点：调用 LLM，可能返回工具调用请求或最终回答。

        所有依赖从 self 读取（llm / system_prompt / max_context_rounds），
        不依赖任何模块级全局变量。
        """
        logger.info("[Node] Agent 节点开始执行")

        with Timer("LLM 调用") as timer:
            llm_with_tools = self._llm.bind_tools(self._tools)

            messages: List[Any] = state.get("messages", [])
            full_messages: List[Any] = [SystemMessage(content=self._system_prompt)]

            recent_messages: List[Any] = safe_truncate_messages(
                messages, self._max_context_rounds
            )
            full_messages.extend(recent_messages)

            response: AIMessage = llm_with_tools.invoke(full_messages)

        llm_time: float = round(timer.elapsed, 4)

        intent: str = "直接回答"
        if response.tool_calls:
            tool_names: List[str] = [
                tc.get("name", "") for tc in response.tool_calls
            ]
            intent = f"调用工具: {', '.join(tool_names)}"

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

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
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

        tool_messages: List[ToolMessage] = []
        tool_results: List[Dict[str, Any]] = []
        retrieved_docs: List[Dict[str, Any]] = []
        monitor_entries: List[Dict[str, Any]] = []

        for tool_call in last_message.tool_calls:
            tool_name: str = tool_call.get("name", "unknown")
            tool_args: Dict[str, Any] = tool_call.get("args", {})
            tool_id: str = tool_call.get("id", "")

            logger.info(f"[Tool] 执行: {tool_name}({list(tool_args.keys())})")

            if tool_name not in self._tools_by_name:
                result: str = f"错误：未知工具 '{tool_name}'"
                logger.error(result)
            else:
                tool_instance = self._tools_by_name[tool_name]
                start_time: float = time.perf_counter()

                try:
                    result = tool_instance._run(**tool_args)
                except Exception as e:
                    result = f"工具执行失败: {e}"
                    logger.error(f"[Tool] {tool_name} 执行异常: {e}", exc_info=True)

                elapsed: float = round(time.perf_counter() - start_time, 4)

                if tool_name == "insurance_rag_search" and hasattr(
                    tool_instance, "_retriever"
                ):
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
                    "success": not str(result).startswith("错误"),
                    "timestamp": time.time(),
                })

                tool_results.append({
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": str(result)[:500],
                    "elapsed": elapsed,
                })

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

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

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

        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)

        builder.set_entry_point("agent")

        builder.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tools",
                "__end__": END,
            },
        )

        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def graph(self):
        """获取编译后的 LangGraph 图。"""
        return self._graph

    @property
    def checkpointer(self) -> InMemorySaver:
        """获取检查点保存器。"""
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
