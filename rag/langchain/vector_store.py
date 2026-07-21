"""
Insurance AI Agent - 向量数据库模块
基于 FAISS 实现本地向量存储和检索。
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from config import VECTOR_STORE_CONFIG
from rag.embedding import EmbeddingManager
from utils.logger import get_logger

logger = get_logger("rag.langchain.vector_store")


class VectorStoreManager:
    """
    FAISS 向量数据库管理器。
    负责索引的构建、保存、加载和检索。
    """

    def __init__(self) -> None:
        """初始化向量库管理器。"""
        self.index_path: Path = Path(VECTOR_STORE_CONFIG["index_path"])
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._embedding_manager = EmbeddingManager()
        self._vector_store: Optional[FAISS] = None

    @property
    def is_loaded(self) -> bool:
        """索引是否已加载到内存。"""
        return self._vector_store is not None

    @property
    def embedding_model_name(self) -> str:
        """获取 Embedding 模型名称。"""
        return self._embedding_manager.model_name

    def index_exists(self) -> bool:
        """
        检查本地是否存在已保存的 FAISS 索引。

        Returns:
            索引文件是否存在
        """
        index_file: Path = self.index_path / "index.faiss"
        return index_file.exists()

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        从 Chunk 列表构建 FAISS 索引并保存到本地。

        Args:
            chunks: 文档 Chunk 列表（来自 DocumentSplitter）
        """
        if not chunks:
            logger.warning("Chunk 列表为空，无法构建索引")
            return

        logger.info(f"开始构建 FAISS 索引，Chunk 数量: {len(chunks)}")

        # 转换为 LangChain Document 格式
        lc_documents: List[LCDocument] = [
            LCDocument(page_content=c["page_content"], metadata=c["metadata"])
            for c in chunks
        ]

        # 提取文本用于 Embedding
        texts: List[str] = [c["page_content"] for c in chunks]

        # 构建 FAISS 索引
        self._vector_store = FAISS.from_documents(
            documents=lc_documents,
            embedding=self._embedding_manager.model,
        )

        # 保存到本地
        self._save_index()

        logger.info(f"FAISS 索引构建完成，已保存到 {self.index_path}")

    def _save_index(self) -> None:
        """将当前索引保存到本地磁盘。"""
        if self._vector_store is None:
            logger.warning("索引为空，无法保存")
            return

        self._vector_store.save_local(str(self.index_path))
        logger.info(f"FAISS 索引已保存到 {self.index_path}")

    def load_index(self) -> bool:
        """
        从本地磁盘加载 FAISS 索引。

        Returns:
            加载是否成功
        """
        if not self.index_exists():
            logger.warning("本地索引文件不存在，无法加载")
            return False

        try:
            self._vector_store = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self._embedding_manager.model,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                f"FAISS 索引加载成功，共 {self._vector_store.index.ntotal} 条向量"
            )
            return True
        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {e}")
            return False

    def similarity_search(
        self, query: str, top_k: int | None = None, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        执行相似度检索。

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认使用配置值
            score_threshold: 相似度阈值，低于此值的结果将被过滤

        Returns:
            检索结果列表，每项包含文本、元数据和相似度分数
        """
        if self._vector_store is None:
            if not self.load_index():
                logger.error("索引未加载，无法执行检索")
                return []

        top_k = top_k or VECTOR_STORE_CONFIG["top_k"]

        # 执行检索（带分数）
        results: List[tuple] = self._vector_store.similarity_search_with_score(
            query, k=top_k
        )

        # 格式化结果
        formatted_results: List[Dict[str, Any]] = []
        for doc, score in results:
            # FAISS 返回距离，转换为相似度分数 (0~1)
            similarity: float = 1.0 / (1.0 + score)
            similarity = round(similarity, 4)

            # 按阈值过滤
            if similarity < score_threshold:
                continue

            formatted_results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": similarity,
            })

        logger.info(
            f"检索完成: query='{query[:50]}...', "
            f"top_k={top_k}, 返回 {len(formatted_results)} 条结果"
        )
        return formatted_results

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息。

        Returns:
            包含向量数量、索引路径等信息的字典
        """
        stats: Dict[str, Any] = {
            "index_loaded": self.is_loaded,
            "index_exists": self.index_exists(),
            "index_path": str(self.index_path),
            "embedding_model": self.embedding_model_name,
        }

        if self._vector_store is not None:
            stats["total_vectors"] = self._vector_store.index.ntotal
        else:
            stats["total_vectors"] = 0

        return stats

    def delete_index(self) -> None:
        """删除本地 FAISS 索引文件。"""
        index_file: Path = self.index_path / "index.faiss"
        pkl_file: Path = self.index_path / "index.pkl"

        deleted: bool = False
        if index_file.exists():
            index_file.unlink()
            deleted = True
        if pkl_file.exists():
            pkl_file.unlink()
            deleted = True

        self._vector_store = None

        if deleted:
            logger.info("FAISS 索引已删除")
        else:
            logger.info("索引文件不存在，无需删除")
