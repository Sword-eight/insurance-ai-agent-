"""
Insurance AI Agent - LangChain 检索器
实现 BaseRetriever 接口，基于 FAISS 向量检索。
"""

from typing import List, Dict, Any

from rag.base_retriever import BaseRetriever
from rag.langchain.vector_store import VectorStoreManager
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.langchain.retriever")


class LangChainRetriever(BaseRetriever):
    """
    LangChain 检索器。
    封装 FAISS 检索逻辑，提供统一的检索接口。
    实现 BaseRetriever 接口。
    """

    def __init__(self) -> None:
        """初始化检索器。"""
        self._vector_store_manager = VectorStoreManager()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行检索并返回格式化结果。

        Args:
            query: 查询问题
            top_k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            检索结果字典，包含 results 列表和耗时信息
        """
        logger.info(f"检索请求: query='{query[:80]}...', top_k={top_k}")

        # 确保索引已加载
        if not self._vector_store_manager.is_loaded:
            if not self._vector_store_manager.load_index():
                logger.error("索引加载失败，返回空结果")
                return {
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "message": "知识库索引不存在，请先上传 PDF 文档并构建索引。",
                }

        # 执行检索
        with Timer("FAISS 检索") as timer:
            results: List[Dict[str, Any]] = (
                self._vector_store_manager.similarity_search(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
            )

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "retrieval_time": round(timer.elapsed, 4),
            "top_k": top_k,
        }

    def format_retrieval_for_llm(self, retrieval_result: Dict[str, Any]) -> str:
        """
        将检索结果格式化为 LLM 可读的文本。

        Args:
            retrieval_result: retrieve() 方法返回的结果字典

        Returns:
            格式化后的文本字符串
        """
        results: List[Dict[str, Any]] = retrieval_result.get("results", [])
        if not results:
            return "根据当前知识库无法确定。"

        formatted_parts: List[str] = []
        for i, result in enumerate(results, 1):
            metadata: Dict[str, Any] = result.get("metadata", {})
            source: str = metadata.get("source", "未知来源")
            page: int = metadata.get("page", 0)
            chunk_id: int = metadata.get("chunk_id", 0)
            similarity: float = result.get("similarity_score", 0.0)
            content: str = result.get("page_content", "")

            formatted_parts.append(
                f"【来源 {i}】\n"
                f"  文档: {source}\n"
                f"  页码: 第 {page} 页\n"
                f"  Chunk 编号: {chunk_id}\n"
                f"  相似度: {similarity:.2%}\n"
                f"  内容: {content}\n"
            )

        return "\n".join(formatted_parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。

        Returns:
            统计信息字典
        """
        return self._vector_store_manager.get_index_stats()

    def rebuild_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        重建索引。

        Args:
            chunks: 文档 Chunk 列表
        """
        logger.info("开始重建知识库索引...")
        self._vector_store_manager.build_index(chunks)

    def delete_index(self) -> None:
        """删除知识库索引。"""
        self._vector_store_manager.delete_index()
