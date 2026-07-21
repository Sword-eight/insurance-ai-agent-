"""
Insurance AI Agent - 保险知识库检索工具
封装 RAG 检索为 LangChain Tool，供 LangGraph Agent 调用。
根据 config.RAG_ENGINE 自动选择 LangChain 或 LlamaIndex 实现。
"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import get_rag_engine
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

    根据 config.RAG_ENGINE 自动选择底层实现。
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

    # 检索器实例（延迟初始化，根据 RAG_ENGINE 选择）
    _retriever: BaseRetriever | None = None

    def __init__(self, **kwargs: Any) -> None:
        """初始化工具，根据 RAG_ENGINE 创建对应的检索器。"""
        super().__init__(**kwargs)
        self._retriever = self._create_retriever()
        engine = get_rag_engine()
        logger.info(
            f"InsuranceRAGTool 初始化: engine={engine}, "
            f"retriever={type(self._retriever).__name__}"
        )

    @staticmethod
    def _create_retriever() -> BaseRetriever:
        """根据 RAG_ENGINE 创建对应的检索器实例。"""
        engine = get_rag_engine()
        if engine == "langchain":
            from rag.langchain.retriever import LangChainRetriever
            return LangChainRetriever()
        elif engine == "llamaindex":
            from rag.llamaindex.retriever import LlamaIndexRetriever
            return LlamaIndexRetriever()
        else:
            raise ValueError(f"不支持的 RAG_ENGINE: {engine}，可选: langchain, llamaindex")

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
                # 执行检索
                result: Dict[str, Any] = self._retriever.retrieve(
                    query=query, top_k=5
                )

                # 格式化为 LLM 可读文本
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
