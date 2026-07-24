"""
Insurance AI Agent - 保险知识库检索工具
封装 RAG 检索为 LangChain Tool，供 LangGraph Agent 调用。

生命周期：RetrievalService 由 init_services() 创建后注入，Tool 不直接调 Retriever。
"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from services.retrieval_service import RetrievalService
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("tools.rag")


class InsuranceRAGInput(BaseModel):
    """insurance_rag_search 工具的参数 Schema。"""

    query: str = Field(
        description="要在保险知识库中检索的问题或关键词，"
        "如'等待期是多长时间'、'重大疾病定义'、'免责条款'等"
    )


class InsuranceRAGTool(BaseTool):
    """
    保险知识库检索工具。
    委托 RetrievalService 执行检索 + 格式化，Tool 只负责 LangChain 接口适配。

    依赖注入：RetrievalService 由外部创建并注入。
    """

    name: str = "insurance_rag_search"
    description: str = (
        "查询保险知识库，获取保险条款、重大疾病定义、理赔规则、"
        "免责条款、等待期、合同终止条件等信息。"
        "适用于需要查找具体保险条款内容的场景。"
        "输入：查询问题或关键词。"
        "返回：知识库检索结果，包含文档来源、章节和相似度分数。"
    )
    args_schema: Type[BaseModel] = InsuranceRAGInput

    # 检索服务（外部注入）
    _service: RetrievalService

    def __init__(self, service: RetrievalService, **kwargs: Any) -> None:
        """
        Args:
            service: RetrievalService 实例（唯一检索入口）
        """
        super().__init__(**kwargs)
        self._service = service
        logger.info(
            f"InsuranceRAGTool 初始化: "
            f"service.retriever={self._service.retriever_type}"
        )

    def _run(self, query: str) -> str:
        """
        执行保险知识库检索（同步）。

        Args:
            query: 检索查询文本

        Returns:
            格式化后的检索结果文本（LLM 可直接使用）
        """
        logger.info(f"[Tool] insurance_rag_search 被调用: query='{query[:80]}...'")

        try:
            with Timer("insurance_rag_search 总耗时") as timer:
                # 委托 Service 层（唯一检索入口）
                result = self._service.search(query=query)
                formatted: str = self._service.format_for_llm(result)

                logger.info(
                    f"[Tool] insurance_rag_search 完成: "
                    f"返回 {result.total_found} 条结果, "
                    f"耗时 {timer.elapsed:.4f}s"
                )

            return formatted

        except Exception as e:
            error_msg: str = f"知识库检索失败: {e}"
            logger.error(f"[Tool] insurance_rag_search 异常: {e}", exc_info=True)
            return error_msg
