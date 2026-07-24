"""
Insurance AI Agent - LangChain 检索器
实现 BaseRetriever 接口，基于 FAISS 向量检索。

职责：query → RetrievalResult。纯数据访问，不做格式化。
"""

from typing import List, Dict, Any

from rag.base_retriever import (
    BaseRetriever,
    RetrievalDocument,
    RetrievalResult,
)
from rag.langchain.vector_store import VectorStoreManager
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.langchain.retriever")


class LangChainRetriever(BaseRetriever):
    """
    LangChain 检索器——Repository 层。

    封装 FAISS 检索逻辑，将 LangChain 特有的 metadata 格式
    归一化为统一的 RetrievalDocument。

    不包含任何格式化、统计、索引管理逻辑。
    """

    def __init__(self, vector_store_manager: VectorStoreManager | None = None) -> None:
        """
        Args:
            vector_store_manager: 可选的外部 VectorStoreManager 实例。
                                  不传则内部创建（向后兼容）。
        """
        self._vector_store_manager = vector_store_manager or VectorStoreManager()

    # ------------------------------------------------------------------
    # BaseRetriever 接口
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> RetrievalResult:
        """
        执行 FAISS 向量检索，返回归一化的 RetrievalResult。

        Args:
            query:           查询文本
            top_k:           返回的最大文档数
            score_threshold: 最低相似度阈值

        Returns:
            RetrievalResult — 包含归一化文档列表和检索元数据
        """
        logger.info(f"[LangChain] retrieve: query='{query[:80]}...', top_k={top_k}")

        # 确保索引已加载
        if not self._vector_store_manager.is_loaded:
            if not self._vector_store_manager.load_index():
                logger.error("索引加载失败，返回空结果")
                return RetrievalResult(query=query, engine="langchain")

        # 执行 FAISS 检索
        with Timer("FAISS 检索") as timer:
            raw_results: List[Dict[str, Any]] = (
                self._vector_store_manager.similarity_search(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
            )

        # 归一化：LangChain metadata → RetrievalDocument
        documents: List[RetrievalDocument] = []
        for raw in raw_results:
            metadata: Dict[str, Any] = raw.get("metadata", {})
            documents.append(RetrievalDocument(
                content=raw.get("page_content", ""),
                source_name=metadata.get("source", "未知来源"),
                source_page=metadata.get("page", 0),
                similarity_score=raw.get("similarity_score", 0.0),
                engine="langchain",
                raw_metadata=metadata,
            ))

        retrieval_time_ms: float = round(timer.elapsed * 1000, 2)

        logger.info(
            f"[LangChain] retrieve 完成: "
            f"返回 {len(documents)} 条, 耗时 {retrieval_time_ms}ms"
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            total_found=len(documents),
            retrieval_time_ms=retrieval_time_ms,
            engine="langchain",
        )
