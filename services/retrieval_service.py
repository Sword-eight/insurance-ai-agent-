"""
Insurance AI Agent - 知识检索服务
唯一的 RAG 检索入口（Single Entry）。负责检索编排、结果组装、LLM 上下文格式化。

设计原则：
  - 所有检索请求必须经过此 Service（Tool 不得直接调 Retriever）
  - 格式化逻辑统一在此处（不再分散在两个 Retriever 中）
  - 未来扩展点：Rerank / Hybrid Search / MultiQuery / Metadata Filter 在此添加
"""

from typing import List, Optional

from rag.base_retriever import (
    BaseRetriever,
    RetrievalDocument,
    RetrievalResult,
)
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("services.retrieval")


class RetrievalService:
    """
    知识检索服务——Service 层。

    编排检索流程：Repository 调用 → 结果组装 → LLM 上下文格式化。
    未来所有检索增强（Rerank、Hybrid、MultiQuery）在此扩展。
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            retriever: 检索器实例（LangChainRetriever 或 LlamaIndexRetriever）
            top_k:     默认返回的文档数量
        """
        self._retriever = retriever
        self._top_k = top_k

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = 0.0,
    ) -> RetrievalResult:
        """
        执行知识库检索。

        Args:
            query:           查询文本
            top_k:           返回文档数（None 则使用默认值）
            score_threshold: 最低相似度阈值

        Returns:
            RetrievalResult — 包含归一化文档列表和检索元数据
        """
        k = top_k if top_k is not None else self._top_k

        logger.info(f"检索请求: query='{query[:80]}...', top_k={k}")

        with Timer("检索总耗时") as timer:
            result = self._retriever.retrieve(
                query=query,
                top_k=k,
                score_threshold=score_threshold,
            )

        service_time_ms: float = round(timer.elapsed * 1000, 2)

        logger.info(
            f"检索完成: 返回 {result.total_found} 条, "
            f"FAISS={result.retrieval_time_ms}ms, "
            f"总数={service_time_ms}ms"
        )

        # 将 Service 层耗时追加到结果中（覆盖 retrieval_time_ms
        # 为端到端时间，原始 FAISS 耗时可通过 raw_metadata 获取）
        result.retrieval_time_ms = service_time_ms

        return result

    def format_for_llm(self, result: RetrievalResult) -> str:
        """
        将检索结果格式化为 LLM 可读的上下文字符串。

        统一格式化逻辑——不再由各 Retriever 各自实现。
        所有 RetrievalDocument 已经过引擎归一化，Service 只处理统一字段。

        Args:
            result: search() 返回的检索结果

        Returns:
            格式化后的文本（中文，含来源引用和相似度标注）
        """
        if not result.documents:
            return "根据当前知识库无法确定。"

        formatted_parts: List[str] = []
        for i, doc in enumerate(result.documents, 1):
            # 引擎标签（可选，便于调试）
            engine_tag: str = f" [{doc.engine}]" if doc.engine else ""

            # 页码信息（如有）
            page_info: str = f"第 {doc.source_page} 页, " if doc.source_page > 0 else ""

            formatted_parts.append(
                f"【来源 {i}】{engine_tag}\n"
                f"  文档: {doc.source_name}\n"
                f"  {page_info}相似度: {doc.similarity_score:.2%}\n"
                f"  内容: {doc.content}\n"
            )

        return "\n".join(formatted_parts)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    @property
    def retriever_type(self) -> str:
        """检索器类型名称（用于调试面板）。"""
        return type(self._retriever).__name__

    @property
    def top_k(self) -> int:
        """默认 top_k 值。"""
        return self._top_k
