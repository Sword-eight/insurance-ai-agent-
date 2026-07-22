"""
Insurance AI Agent - 保险知识库检索工具
封装 RAG 检索为 LangChain Tool，供 LangGraph Agent 调用。

生命周期：Retriever 由 init_services() 创建后注入，Tool 不自行 new。
"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from rag.base_retriever import BaseRetriever
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
    基于 RAG 管道检索保险条款、理赔规则等信息，
    返回检索到的文本片段及来源引用。

    依赖注入：Retriever 由外部创建并注入，Tool 不关心底层实现。
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

    # 检索器实例（外部注入）
    _retriever: BaseRetriever

    def __init__(self, retriever: BaseRetriever, **kwargs: Any) -> None:
        """
        Args:
            retriever: 检索器实例（LangChainRetriever 或 LlamaIndexRetriever）
        """
        super().__init__(**kwargs)
        self._retriever = retriever
        logger.info(
            f"InsuranceRAGTool 初始化: retriever={type(self._retriever).__name__}"
        )

    def _run(self, query: str) -> str:
        """
        执行保险知识库检索（同步）。

        Args:
            query: 检索查询文本

        Returns:
            格式化后的检索结果文本
        """
        logger.info(f"[Tool] insurance_rag_search 被调用: query='{query[:80]}...'")

        try:
            with Timer("insurance_rag_search 总耗时") as timer:
                result: Dict[str, Any] = self._retriever.retrieve(
                    query=query, top_k=5
                )

                formatted: str = self._retriever.format_retrieval_for_llm(result)

                logger.info(
                    f"[Tool] insurance_rag_search 完成: "
                    f"返回 {result['total_results']} 条结果, "
                    f"耗时 {timer.elapsed:.4f}s"
                )

            return formatted

        except Exception as e:
            error_msg: str = f"知识库检索失败: {e}"
            logger.error(f"[Tool] insurance_rag_search 异常: {e}", exc_info=True)
            return error_msg
