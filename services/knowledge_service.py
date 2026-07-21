"""
Insurance AI Agent - 知识库服务模块
编排 PDF 加载、文档切分、索引构建的完整流程。
支持 LangChain 和 LlamaIndex 两套 RAG 引擎。
"""

from pathlib import Path
from typing import List, Dict, Any

from config import get_rag_engine, PDF_CONFIG
from rag.base_retriever import BaseRetriever
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("services.knowledge")


class KnowledgeService:
    """
    知识库管理服务。
    编排完整的 RAG 管道：加载 → 切分 → 索引。
    对外提供知识库构建、重建、状态查询等功能。

    根据 RAG_ENGINE 配置自动选择底层实现：
      - "langchain":   PyMuPDF → RecursiveCharacterTextSplitter → FAISS
      - "llamaindex":  SimpleDirectoryReader → SentenceSplitter → VectorStoreIndex → FAISS
    """

    def __init__(self) -> None:
        """初始化知识库服务，根据 RAG_ENGINE 创建对应组件。"""
        self._rag_engine: str = get_rag_engine()

        if self._rag_engine == "langchain":
            self._init_langchain()
        elif self._rag_engine == "llamaindex":
            self._init_llamaindex()
        else:
            raise ValueError(f"不支持的 RAG_ENGINE: {self._rag_engine}，可选: langchain, llamaindex")

        logger.info(f"KnowledgeService 初始化完成，引擎: {self._rag_engine}")

    def _init_langchain(self) -> None:
        """初始化 LangChain RAG 组件。"""
        from rag.langchain.loader import PDFLoader
        from rag.langchain.splitter import DocumentSplitter
        from rag.langchain.vector_store import VectorStoreManager
        from rag.langchain.retriever import LangChainRetriever

        self._loader = PDFLoader()
        self._splitter = DocumentSplitter()
        self._vector_store = VectorStoreManager()
        self._retriever: BaseRetriever = LangChainRetriever()
        self._builder = None  # LangChain 不使用 builder

    def _init_llamaindex(self) -> None:
        """初始化 LlamaIndex RAG 组件。"""
        from rag.llamaindex.index_builder import LlamaIndexBuilder
        from rag.llamaindex.retriever import LlamaIndexRetriever

        self._builder = LlamaIndexBuilder()
        self._retriever: BaseRetriever = LlamaIndexRetriever()
        self._loader = None
        self._splitter = None
        self._vector_store = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def get_retriever(self) -> BaseRetriever:
        """获取当前引擎的检索器实例。"""
        return self._retriever

    @property
    def rag_engine(self) -> str:
        """获取当前 RAG 引擎名称。"""
        return self._rag_engine

    # ------------------------------------------------------------------
    # 知识库构建
    # ------------------------------------------------------------------

    def build_knowledge_base(self) -> Dict[str, Any]:
        """
        构建知识库索引。
        如果本地已有索引则加载，否则自动扫描文档并构建。

        Returns:
            构建结果，包含 PDF 数量、Chunk 数量、耗时信息
        """
        if self._rag_engine == "langchain":
            return self._build_langchain()
        elif self._rag_engine == "llamaindex":
            return self._build_llamaindex()
        else:
            return {"success": False, "message": f"不支持的引擎: {self._rag_engine}"}

    def _build_langchain(self) -> Dict[str, Any]:
        """LangChain 流程：PDFLoader → DocumentSplitter → VectorStoreManager。"""
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

    def _build_llamaindex(self) -> Dict[str, Any]:
        """LlamaIndex 流程：SimpleDirectoryReader → SentenceSplitter → VectorStoreIndex。"""
        from llama_index.core import SimpleDirectoryReader

        result: Dict[str, Any] = {
            "success": False,
            "pdf_count": 0,
            "chunk_count": 0,
            "vector_count": 0,
            "message": "",
        }

        try:
            # 检查文档
            pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
            doc_files: list = list(pdf_dir.glob("*.*"))
            doc_files = [f for f in doc_files if f.suffix.lower() in (".pdf", ".txt")]
            result["pdf_count"] = len(doc_files)

            if not doc_files:
                result["message"] = "data/pdf/ 目录中没有文档文件，请先上传文档"
                logger.warning(result["message"])
                return result

            # 使用 SimpleDirectoryReader 加载
            with Timer("LlamaIndex 文档加载"):
                documents = SimpleDirectoryReader(
                    input_dir=str(pdf_dir),
                    recursive=False,
                ).load_data()

            if not documents:
                result["message"] = "文档加载失败或文档内容为空"
                return result

            # 构建索引（内部会做 SentenceSplitter 分块 + FAISS 索引）
            with Timer("LlamaIndex 索引构建"):
                self._builder.build_index(documents=documents)

            result.update({
                "success": True,
                "chunk_count": len(documents),  # LlamaIndex 在 build 时统计
                "message": f"LlamaIndex 知识库构建成功: {len(doc_files)} 个文档",
            })
            logger.info(result["message"])

        except Exception as e:
            result["message"] = f"LlamaIndex 知识库构建失败: {e}"
            logger.error(result["message"], exc_info=True)

        return result

    def load_existing_index(self) -> bool:
        """
        加载已存在的本地索引。

        Returns:
            是否加载成功
        """
        if self._rag_engine == "langchain":
            return self._vector_store.load_index()
        elif self._rag_engine == "llamaindex":
            return self._builder.load_index()
        return False

    # ------------------------------------------------------------------
    # 统计与查询
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。

        Returns:
            统计信息字典
        """
        if self._rag_engine == "langchain":
            stats = self._vector_store.get_index_stats()
            stats["pdf_count"] = self._loader.get_pdf_count()
            stats["chunk_size"] = self._splitter.chunk_size
            stats["chunk_overlap"] = self._splitter.chunk_overlap
        elif self._rag_engine == "llamaindex":
            stats = self._builder.get_stats()
            pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
            doc_files: list = list(pdf_dir.glob("*.*")) if pdf_dir.exists() else []
            doc_files = [f for f in doc_files if f.suffix.lower() in (".pdf", ".txt")]
            stats["pdf_count"] = len(doc_files)
            stats["chunk_size"] = PDF_CONFIG["chunk_size"]
            stats["chunk_overlap"] = PDF_CONFIG["chunk_overlap"]
        else:
            stats = {}

        stats["rag_engine"] = self._rag_engine
        return stats

    def get_pdf_files(self) -> List[Dict[str, Any]]:
        """
        获取文档文件列表。

        Returns:
            文档文件信息列表
        """
        pdf_dir: Path = Path(PDF_CONFIG["pdf_dir"])
        if not pdf_dir.exists():
            return []

        if self._rag_engine == "langchain" and self._loader is not None:
            pdf_files: List[Path] = self._loader.list_pdfs()
        else:
            pdf_files = sorted(
                [f for f in pdf_dir.glob("*.*")
                 if f.suffix.lower() in (".pdf", ".txt")]
            )

        return [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "path": str(f),
            }
            for f in pdf_files
        ]

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------

    def rebuild(self) -> Dict[str, Any]:
        """
        重建知识库——删除旧索引后重新构建。

        Returns:
            重建结果
        """
        logger.info(f"开始重建知识库（引擎: {self._rag_engine}）...")
        self.delete_knowledge_base()
        return self.build_knowledge_base()

    def delete_knowledge_base(self) -> Dict[str, Any]:
        """
        删除知识库索引。

        Returns:
            操作结果
        """
        if self._rag_engine == "langchain":
            self._vector_store.delete_index()
        elif self._rag_engine == "llamaindex":
            self._builder.delete_index()

        return {
            "success": True,
            "message": "知识库索引已删除",
        }
