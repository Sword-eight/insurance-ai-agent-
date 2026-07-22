"""
事件处理模块。
所有用户操作（发送消息、上传 PDF、重建索引等）的处理逻辑。

纯逻辑函数，不包含任何 st.xxx() UI 渲染调用。
返回结构化结果，由 UI 层负责渲染。
"""

import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage

from config import PDF_CONFIG
from utils.logger import get_logger

logger = get_logger("app.handlers")


# ==================================================================
# 聊天消息处理
# ==================================================================

def handle_chat_message(
    prompt: str,
    graph_builder: Any,
    session_id: str,
) -> Dict[str, Any]:
    """
    处理用户聊天消息：调用 Agent → 解析结果。

    Args:
        prompt: 用户输入文本
        graph_builder: AgentGraphBuilder 实例
        session_id: 当前会话 ID

    Returns:
        {
            "final_answer": str,
            "tool_calls": List[Dict],
            "retrieved_sources": List[Dict],
            "agent_monitor": List[Dict],
            "total_time": float,
        }
    """
    total_start: float = time.perf_counter()

    result = graph_builder.invoke(
        user_message=prompt,
        session_id=session_id,
    )

    total_time: float = round(time.perf_counter() - total_start, 2)

    return _parse_agent_result(result, total_time)


def _parse_agent_result(
    result: Dict[str, Any],
    total_time: float,
) -> Dict[str, Any]:
    """
    解析 Agent 返回的原始结果，提取最终回答、工具调用、RAG 来源、监控数据。

    Args:
        result: graph_builder.invoke() 的返回值
        total_time: 总耗时

    Returns:
        结构化的解析结果
    """
    messages = result.get("messages", [])

    # 提取最终回答
    final_answer: str = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    if not final_answer:
        final_answer = "抱歉，我无法处理您的请求，请稍后重试。"

    # 收集 Tool 调用信息
    tool_calls_info: List[Dict] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_info.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })

    # 收集 Agent Monitor
    monitor_entries: List[Dict] = result.get("agent_monitor", [])
    for entry in monitor_entries:
        entry["total_time"] = total_time

    # 收集 RAG 检索来源
    retrieved_sources: List[Dict] = []
    tool_results: List[Dict] = result.get("tool_results", [])
    for tr in tool_results:
        if tr.get("tool_name") == "insurance_rag_search":
            result_text: str = tr.get("result", "")
            engine_label: str = (
                "LlamaIndex" if "引擎: LlamaIndex" in result_text
                else "LangChain"
            )
            retrieved_sources.append({
                "source": f"知识库检索结果 ({engine_label})",
                "engine": engine_label,
                "similarity": "N/A",
                "content": result_text[:800],
            })

    logger.info(
        f"聊天处理完成: total_time={total_time}s, "
        f"tool_calls={len(tool_calls_info)}, sources={len(retrieved_sources)}"
    )

    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls_info,
        "retrieved_sources": retrieved_sources,
        "agent_monitor": monitor_entries,
        "total_time": total_time,
    }


# ==================================================================
# 知识库管理处理
# ==================================================================

def handle_pdf_upload(
    uploaded_files: List[Any],
    knowledge_service: Any,
) -> Dict[str, Any]:
    """
    处理 PDF 上传：保存文件 → 重建索引。

    Args:
        uploaded_files: Streamlit UploadedFile 对象列表
        knowledge_service: KnowledgeService 实例

    Returns:
        {"success": bool, "saved_count": int, "message": str}
    """
    saved_count: int = 0
    for uploaded_file in uploaded_files:
        save_path: Path = Path(PDF_CONFIG["pdf_dir"]) / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_count += 1

    result = knowledge_service.rebuild()

    logger.info(f"PDF 上传完成: {saved_count} 个文件, {result['message']}")

    return {
        "success": result["success"],
        "saved_count": saved_count,
        "message": result["message"],
    }


def handle_rebuild_index(knowledge_service: Any) -> Dict[str, Any]:
    """
    处理索引重建。

    Args:
        knowledge_service: KnowledgeService 实例

    Returns:
        {"success": bool, "message": str}
    """
    result = knowledge_service.rebuild()
    logger.info(f"索引重建: {result['message']}")
    return {"success": result["success"], "message": result["message"]}


def handle_delete_index(knowledge_service: Any) -> Dict[str, Any]:
    """
    处理索引删除。

    Args:
        knowledge_service: KnowledgeService 实例

    Returns:
        {"success": bool, "message": str}
    """
    result = knowledge_service.delete_knowledge_base()
    logger.info(f"索引删除: {result['message']}")
    return {"success": result["success"], "message": result["message"]}
