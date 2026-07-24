"""
Insurance AI Agent - LlamaIndex 检索器
实现 BaseRetriever 接口，使用 LlamaIndex as_retriever() 进行检索。

职责：query → RetrievalResult。纯数据访问，不做格式化。
不调用 LLM，不生成回答。
"""

from typing import List, Dict, Any

from rag.base_retriever import (
    BaseRetriever,
    RetrievalDocument,
    RetrievalResult,
)
from rag.llamaindex.index_builder import LlamaIndexBuilder
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.llamaindex.retriever")


def _extract_chunk_order(node: Any) -> int:
    """从 Node relationships 推断 chunk 在文档中的顺序位置。"""
    try:
        if hasattr(node, 'relationships') and node.relationships:
            from llama_index.core.schema import NodeRelationship
            has_prev = NodeRelationship.PREVIOUS in node.relationships
            has_next = NodeRelationship.NEXT in node.relationships
            if has_prev and has_next:
                return -1
            elif not has_prev and has_next:
                return 1
            elif has_prev and not has_next:
                return -2
            elif not has_prev and not has_next:
                return 0
    except Exception:
        pass
    return -1


class LlamaIndexRetriever(BaseRetriever):
    """
    LlamaIndex 检索器——Repository 层。

    使用 LlamaIndex retriever 从 FAISS 索引中获取 Node 列表，
    将 LlamaIndex 特有的 metadata/relationships 归一化为统一的 RetrievalDocument。

    不调用 LLM，不包含任何格式化逻辑。
    """

    def __init__(self, builder: LlamaIndexBuilder | None = None) -> None:
        """
        Args:
            builder: 可选的外部 LlamaIndexBuilder 实例。
                     不传则内部创建（向后兼容）。
        """
        self._builder = builder or LlamaIndexBuilder()

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
        执行 LlamaIndex 检索，返回归一化的 RetrievalResult。

        Args:
            query:           查询文本
            top_k:           返回的最大文档数
            score_threshold: 最低相似度阈值

        Returns:
            RetrievalResult
        """
        logger.info(f"[LlamaIndex] retrieve: query='{query[:80]}...', top_k={top_k}")

        if not self._builder.is_loaded:
            if not self._builder.load_index():
                logger.error("LlamaIndex 索引加载失败，返回空结果")
                return RetrievalResult(query=query, engine="llamaindex")

        with Timer("LlamaIndex 检索") as timer:
            retriever = self._builder.index.as_retriever(
                similarity_top_k=top_k,
            )
            nodes: list = retriever.retrieve(query)

        # 归一化：LlamaIndex Node → RetrievalDocument
        documents: List[RetrievalDocument] = []
        for node in nodes:
            raw_score: float = node.score or 0.0
            similarity: float = round(max(0.0, min(1.0, raw_score)), 4)

            if similarity < score_threshold:
                continue

            metadata: Dict[str, Any] = dict(node.metadata) if node.metadata else {}

            # 归一化字段映射
            source_name: str = metadata.get("file_name", "未知来源")
            source_page: int = 0  # LlamaIndex 默认无页码
            chunk_order: int = _extract_chunk_order(
                node.node if hasattr(node, 'node') else node
            )

            documents.append(RetrievalDocument(
                content=node.text or node.get_content(),
                source_name=source_name,
                source_page=source_page,
                similarity_score=similarity,
                engine="llamaindex",
                raw_metadata={
                    **metadata,
                    "node_id": node.node_id,
                    "chunk_order": chunk_order,
                },
            ))

        retrieval_time_ms: float = round(timer.elapsed * 1000, 2)

        logger.info(
            f"[LlamaIndex] retrieve 完成: "
            f"返回 {len(documents)} 条, 耗时 {retrieval_time_ms}ms"
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            total_found=len(documents),
            retrieval_time_ms=retrieval_time_ms,
            engine="llamaindex",
        )
