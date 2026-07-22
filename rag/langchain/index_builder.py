"""
Insurance AI Agent - LangChain 索引构建器
包装 PDFLoader + DocumentSplitter + VectorStoreManager，
实现 BaseIndexBuilder 接口。
"""

from pathlib import Path
from typing import List, Dict, Any

from rag.base_index_builder import BaseIndexBuilder
from rag.langchain.loader import PDFLoader
from rag.langchain.splitter import DocumentSplitter
from rag.langchain.vector_store import VectorStoreManager
from config import PDF_CONFIG
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.langchain.index_builder")


class LangChainIndexBuilder(BaseIndexBuilder):
    """
    LangChain 索引构建器。

    流程：PDFLoader → DocumentSplitter → VectorStoreManager(FAISS)
    实现 BaseIndexBuilder 接口，对 KnowledgeService 隐藏引擎细节。

    所有依赖通过构造函数注入（loader / splitter / vector_store 可替换）。
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        loader: PDFLoader | None = None,
        splitter: DocumentSplitter | None = None,
    ) -> None:
        """
        Args:
            vector_store: FAISS 向量库管理器
            loader:       PDF 文档加载器（默认创建）
            splitter:     文档切分器（默认创建）
        """
        self._vector_store = vector_store
        self._loader = loader or PDFLoader()
        self._splitter = splitter or DocumentSplitter()

    # ------------------------------------------------------------------
    # BaseIndexBuilder 接口实现
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """
        构建索引：扫描文档 → 加载 → 切分 → 向量化 → 保存。

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
            pdf_count: int = self._loader.get_pdf_count()
            result["pdf_count"] = pdf_count

            if pdf_count == 0:
                result["message"] = "data/pdf/ 目录中没有文档文件，请先上传文档"
                logger.warning(result["message"])
                return result

            with Timer("PDF 加载"):
                documents: List[Dict[str, Any]] = self._loader.load_all_pdfs()

            if not documents:
                result["message"] = "文档加载失败或文档内容为空"
                return result

            with Timer("文档切分"):
                chunks: List[Dict[str, Any]] = self._splitter.split_documents(documents)
            result["chunk_count"] = len(chunks)

            with Timer("索引构建"):
                self._vector_store.build_index(chunks)

            stats = self._vector_store.get_index_stats()
            result.update({
                "success": True,
                "vector_count": stats.get("total_vectors", 0),
                "message": (
                    f"知识库构建成功: {pdf_count} 个文档 → "
                    f"{len(chunks)} 个 Chunk → "
                    f"{stats.get('total_vectors', 0)} 条向量"
                ),
            })
            logger.info(result["message"])

        except Exception as e:
            result["message"] = f"知识库构建失败: {e}"
            logger.error(result["message"], exc_info=True)

        return result

    def load(self) -> bool:
        """从磁盘加载已有 FAISS 索引。"""
        return self._vector_store.load_index()

    def delete(self) -> None:
        """删除 FAISS 索引文件。"""
        self._vector_store.delete_index()

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计 + 文档信息。"""
        stats = self._vector_store.get_index_stats()
        stats["pdf_count"] = self._loader.get_pdf_count()
        stats["chunk_size"] = self._splitter.chunk_size
        stats["chunk_overlap"] = self._splitter.chunk_overlap
        stats["rag_engine"] = "langchain"
        return stats

    def index_exists(self) -> bool:
        """检查 FAISS 索引文件是否存在。"""
        return self._vector_store.index_exists()

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出文档目录中的所有文件。"""
        pdf_files: List[Path] = self._loader.list_pdfs()
        return [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "path": str(f),
            }
            for f in pdf_files
        ]
