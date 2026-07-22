"""
Insurance AI Agent - LlamaIndex 索引构建器
使用 LlamaIndex 管理 FAISS 索引的完整生命周期。
实现 BaseIndexBuilder 接口。

注意：
  - 不使用 QueryEngine，不调用 LLM。
  - 仅负责索引的生命周期管理。
  - Embedding 复用项目统一的 SentenceTransformer。
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

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

from rag.base_index_builder import BaseIndexBuilder
from config import PDF_CONFIG, VECTOR_STORE_CONFIG
from rag.embedding import LlamaIndexEmbeddingAdapter
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.llamaindex.builder")


class LlamaIndexBuilder(BaseIndexBuilder):
    """
    LlamaIndex 索引构建器。

    使用 LlamaIndex 框架管理 FAISS 索引的完整生命周期：
    构建 → 保存 → 加载 → 删除。

    注意：LlamaIndex 使用独立的索引目录，与 LangChain 索引隔离。
    """

    def __init__(self) -> None:
        """初始化组件。"""
        self._index_path: Path = Path(VECTOR_STORE_CONFIG["index_path"]) / "llamaindex"
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._embedding = LlamaIndexEmbeddingAdapter()
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
    # BaseIndexBuilder 接口 — build / load / delete
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """
        构建索引：自动扫描文档目录 → 加载 → 切分 → 向量化 → 保存。

        Returns:
            构建结果
        """
        result: Dict[str, Any] = {
            "success": False,
            "pdf_count": 0,
            "chunk_count": 0,
            "vector_count": 0,
            "message": "",
        }

        try:
            pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
            doc_files: list = list(pdf_dir.glob("*.*"))
            doc_files = [f for f in doc_files if f.suffix.lower() in (".pdf", ".txt")]
            result["pdf_count"] = len(doc_files)

            if not doc_files:
                result["message"] = "data/pdf/ 目录中没有文档文件，请先上传文档"
                logger.warning(result["message"])
                return result

            with Timer("LlamaIndex 文档加载"):
                documents = SimpleDirectoryReader(
                    input_dir=str(pdf_dir),
                    recursive=False,
                ).load_data()

            if not documents:
                result["message"] = "文档加载失败或文档内容为空"
                return result

            with Timer("LlamaIndex 索引构建"):
                self._build_index(documents=documents)

            result.update({
                "success": True,
                "chunk_count": len(documents),
                "message": f"LlamaIndex 知识库构建成功: {len(doc_files)} 个文档",
            })
            logger.info(result["message"])

        except Exception as e:
            result["message"] = f"LlamaIndex 知识库构建失败: {e}"
            logger.error(result["message"], exc_info=True)

        return result

    def load(self) -> bool:
        """从磁盘加载索引。"""
        if not self.index_exists():
            logger.warning("本地索引文件不存在，无法加载")
            return False

        try:
            from llama_index.core import load_index_from_storage

            Settings.embed_model = self._embedding
            Settings.llm = None

            faiss_store = FaissVectorStore.from_persist_dir(str(self._index_path))
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

    def delete(self) -> None:
        """删除索引目录。"""
        self._index = None
        if self._index_path.exists():
            shutil.rmtree(str(self._index_path))
            logger.info(f"LlamaIndex 索引已删除: {self._index_path}")
        else:
            logger.info("LlamaIndex 索引目录不存在，无需删除")

    # ------------------------------------------------------------------
    # BaseIndexBuilder 接口 — 查询
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息。"""
        pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
        doc_files: list = list(pdf_dir.glob("*.*")) if pdf_dir.exists() else []
        doc_files = [f for f in doc_files if f.suffix.lower() in (".pdf", ".txt")]

        return {
            "index_loaded": self.is_loaded,
            "index_exists": self.index_exists(),
            "index_path": str(self._index_path),
            "embedding_model": "BAAI/bge-base-zh-v1.5 (via SentenceTransformer)",
            "total_vectors": 0,  # LlamaIndex 不直接暴露向量数
            "pdf_count": len(doc_files),
            "chunk_size": PDF_CONFIG["chunk_size"],
            "chunk_overlap": PDF_CONFIG["chunk_overlap"],
            "rag_engine": "llamaindex",
        }

    def index_exists(self) -> bool:
        """检查本地是否存在已保存的索引。"""
        docstore = self._index_path / "docstore.json"
        return docstore.exists()

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出文档目录中的文件。"""
        pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
        if not pdf_dir.exists():
            return []

        files = sorted(
            [f for f in pdf_dir.glob("*.*")
             if f.suffix.lower() in (".pdf", ".txt")]
        )
        return [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "path": str(f),
            }
            for f in files
        ]

    # ------------------------------------------------------------------
    # 内部方法（保留向后兼容）
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[Any]] = None,
    ) -> None:
        """
        构建 FAISS 索引并持久化（内部方法，也可从外部直接调用用于测试）。

        Args:
            documents: LlamaIndex Document 列表
            nodes:     预处理好的 Node 列表（与 documents 二选一）
        """
        self._build_index(documents=documents, nodes=nodes)

    def _build_index(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[Any]] = None,
    ) -> None:
        """内部索引构建逻辑。"""
        Settings.embed_model = self._embedding
        Settings.llm = None

        if self.index_exists():
            self.delete()

        faiss_dim: int = len(self._embedding._get_query_embedding("dim_test"))
        faiss_index = faiss.IndexFlatL2(faiss_dim)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(vector_store=faiss_store)

        if documents:
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

        self._save_index()
        logger.info(f"LlamaIndex 索引构建完成，已保存到 {self._index_path}")

    def _save_index(self) -> None:
        """将索引持久化到磁盘。"""
        if self._index is None:
            logger.warning("索引为空，无法保存")
            return

        self._index.storage_context.persist(persist_dir=str(self._index_path))

        vector_store = self._index.storage_context.vector_store
        if hasattr(vector_store, '_faiss_index') and vector_store._faiss_index is not None:
            faiss_index_path = self._index_path / "faiss.index"
            faiss.write_index(vector_store._faiss_index, str(faiss_index_path))

        logger.info(f"LlamaIndex 索引已保存到 {self._index_path}")
