"""
Insurance AI Agent - LlamaIndex 索引构建器
使用 LlamaIndex 管理 FAISS 索引的构建、保存、加载和删除。

注意：
  - 不使用 QueryEngine，不调用 LLM。
  - 仅负责索引的生命周期管理。
  - Embedding 复用项目统一的 SentenceTransformer。
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Any

import faiss
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from config import PDF_CONFIG, VECTOR_STORE_CONFIG
from rag.embedding import LlamaIndexEmbeddingAdapter
from utils.logger import get_logger

logger = get_logger("rag.llamaindex.builder")


class LlamaIndexBuilder:
    """
    LlamaIndex 索引构建器。

    使用 LlamaIndex 框架管理 FAISS 索引的完整生命周期：
    构建 → 保存 → 加载 → 删除。

    注意：LlamaIndex 使用独立的索引目录，与 LangChain 索引隔离。
    """

    def __init__(self) -> None:
        """初始化组件。"""
        # LlamaIndex 使用独立索引路径，与 LangChain 索引隔离
        self._index_path: Path = Path(VECTOR_STORE_CONFIG["index_path"]) / "llamaindex"
        self._index_path.mkdir(parents=True, exist_ok=True)

        # 共享 Embedding（通过适配器复用项目统一模型）
        self._embedding = LlamaIndexEmbeddingAdapter()

        # 索引实例（None 表示未加载）
        self._index: Optional[VectorStoreIndex] = None

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """获取当前加载的索引实例。"""
        return self._index

    @property
    def is_loaded(self) -> bool:
        """索引是否已加载到内存。"""
        return self._index is not None

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[Any]] = None,
    ) -> None:
        """
        构建 FAISS 索引并持久化。

        Args:
            documents: LlamaIndex Document 列表（由 SimpleDirectoryReader 产生）
            nodes: 预处理好的 Node 列表（与 documents 二选一）
        """
        # 配置全局 Settings（仅影响此次构建，不污染全局状态）
        Settings.embed_model = self._embedding
        Settings.llm = None  # 不调用 LLM

        # 创建 FAISS 向量库（始终新建，不复用旧索引，避免 ID 不一致）
        # 先删除旧索引文件，确保干净的构建环境
        if self.index_exists():
            self.delete_index()
        # 从 EmbeddingManager 获取向量维度
        faiss_dim: int = len(self._embedding._get_query_embedding("dim_test"))
        faiss_index = faiss.IndexFlatL2(faiss_dim)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(
            vector_store=faiss_store,
        )

        if documents:
            # 使用内置 SentenceSplitter 做 Node Parser
            node_parser = SentenceSplitter(
                chunk_size=PDF_CONFIG["chunk_size"],
                chunk_overlap=PDF_CONFIG["chunk_overlap"],
            )

            logger.info(
                f"开始构建 LlamaIndex 索引: {len(documents)} 个文档, "
                f"chunk_size={PDF_CONFIG['chunk_size']}, "
                f"chunk_overlap={PDF_CONFIG['chunk_overlap']}"
            )

            self._index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=self._embedding,
                transformations=[node_parser],
                show_progress=False,
            )
        elif nodes:
            logger.info(f"开始构建 LlamaIndex 索引: {len(nodes)} 个 Node")
            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self._embedding,
                show_progress=False,
            )
        else:
            logger.warning("documents 和 nodes 均为空，无法构建索引")
            return

        # 持久化到磁盘
        self._save_index()
        logger.info(f"LlamaIndex 索引构建完成，已保存到 {self._index_path}")

    def _save_index(self) -> None:
        """将索引持久化到磁盘。"""
        if self._index is None:
            logger.warning("索引为空，无法保存")
            return

        # 1. 保存 storage context（docstore, index_store, graph_store 等）
        self._index.storage_context.persist(persist_dir=str(self._index_path))

        # 2. 单独保存 FAISS 索引二进制文件
        vector_store = self._index.storage_context.vector_store
        if hasattr(vector_store, '_faiss_index') and vector_store._faiss_index is not None:
            faiss_index_path = self._index_path / "faiss.index"
            faiss.write_index(vector_store._faiss_index, str(faiss_index_path))

        logger.info(f"LlamaIndex 索引已保存到 {self._index_path}")

    # ------------------------------------------------------------------
    # 索引加载 / 检测 / 删除
    # ------------------------------------------------------------------

    def index_exists(self) -> bool:
        """检查本地是否存在已保存的索引。"""
        # LlamaIndex 持久化后会产生 docstore.json 等文件
        docstore = self._index_path / "docstore.json"
        return docstore.exists()

    def load_index(self) -> bool:
        """从磁盘加载索引。"""
        if not self.index_exists():
            logger.warning("本地索引文件不存在，无法加载")
            return False

        try:
            from llama_index.core import load_index_from_storage

            Settings.embed_model = self._embedding
            Settings.llm = None

            # 重建 FAISS vector store 从持久化目录
            faiss_store = FaissVectorStore.from_persist_dir(
                str(self._index_path)
            )
            storage_context = StorageContext.from_defaults(
                vector_store=faiss_store,
                persist_dir=str(self._index_path),
            )

            self._index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self._embedding,
            )

            logger.info("LlamaIndex 索引加载成功")
            return True

        except Exception as e:
            logger.error(f"加载 LlamaIndex 索引失败: {e}", exc_info=True)
            return False

    def delete_index(self) -> None:
        """删除索引目录。"""
        self._index = None
        if self._index_path.exists():
            shutil.rmtree(str(self._index_path))
            logger.info(f"LlamaIndex 索引已删除: {self._index_path}")
        else:
            logger.info("LlamaIndex 索引目录不存在，无需删除")

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """获取索引统计信息。"""
        return {
            "index_loaded": self.is_loaded,
            "index_exists": self.index_exists(),
            "index_path": str(self._index_path),
            "embedding_model": "BAAI/bge-base-zh-v1.5 (via SentenceTransformer)",
        }
