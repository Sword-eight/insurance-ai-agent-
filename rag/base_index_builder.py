"""
Insurance AI Agent - 索引构建器抽象基类
定义统一的索引构建接口，与 BaseRetriever 对等。

所有 RAG 实现（LangChain、LlamaIndex 等）必须实现此接口，
确保 KnowledgeService 无需感知具体引擎。

设计原则（开闭原则）：
    对扩展开放 — 新增第三种 RAG 实现只需继承此基类。
    对修改关闭 — KnowledgeService 只依赖此抽象，不包含任何 if/elif。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseIndexBuilder(ABC):
    """
    索引构建器抽象基类。

    所有 RAG 实现（LangChainIndexBuilder / LlamaIndexBuilder / ...）
    必须实现此接口的全部方法。

    KnowledgeService 只持有 BaseIndexBuilder 引用，
    通过 build() / load() / delete() / get_stats() / index_exists() / list_documents()
    六个方法完成所有操作，不包含任何引擎分支。
    """

    @abstractmethod
    def build(self) -> Dict[str, Any]:
        """
        构建索引（扫描文档 → 切分 → 向量化 → 保存）。

        Returns:
            构建结果，必须包含以下键：
            - success: bool
            - pdf_count: int
            - message: str
        """
        ...

    @abstractmethod
    def load(self) -> bool:
        """
        从磁盘加载已有索引。

        Returns:
            加载是否成功
        """
        ...

    @abstractmethod
    def delete(self) -> None:
        """删除当前索引（包括磁盘文件）。"""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息。

        Returns:
            统计信息字典，包含但不限于：
            - index_loaded: bool
            - index_exists: bool
            - total_vectors: int
            - embedding_model: str
            - pdf_count: int
            - chunk_size: int
            - chunk_overlap: int
            - rag_engine: str
        """
        ...

    @abstractmethod
    def index_exists(self) -> bool:
        """
        检查本地是否存在已保存的索引。

        Returns:
            索引是否存在
        """
        ...

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出可用的文档文件。

        Returns:
            文档信息列表，每项包含 name, size, path
        """
        ...
