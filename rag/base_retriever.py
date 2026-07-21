"""
Insurance AI Agent - 检索器抽象基类
定义统一检索接口，所有 RAG 实现（LangChain、LlamaIndex 等）必须实现此接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
    """
    检索器抽象基类。

    所有 RAG 实现（LangChainRetriever / LlamaIndexRetriever / ...）
    必须实现此接口的全部方法，确保上层调用方无需感知具体实现。

    设计原则（开闭原则）：
        对扩展开放 — 新增第三种 RAG 实现只需继承此基类。
        对修改关闭 — KnowledgeService / InsuranceRAGTool 只依赖此抽象。
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行检索并返回结构化结果。

        Args:
            query: 查询问题
            top_k: 返回结果数量
            score_threshold: 相似度阈值（低于此值的结果将被过滤）

        Returns:
            检索结果字典，必须包含以下键：
            - query: str          原始查询
            - results: List[Dict] 检索结果列表（每项含 page_content, metadata, similarity_score）
            - total_results: int  结果数量
            - retrieval_time: float  检索耗时（秒）
            - top_k: int          请求的结果数量
        """
        ...

    @abstractmethod
    def format_retrieval_for_llm(self, retrieval_result: Dict[str, Any]) -> str:
        """
        将检索结果格式化为 LLM 可读的文本。

        Args:
            retrieval_result: retrieve() 方法返回的结果字典

        Returns:
            格式化后的文本字符串（中文，含来源引用和相似度标注）
        """
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。

        Returns:
            统计信息字典，包含但不限于：
            - index_loaded / index_exists
            - total_vectors
            - embedding_model
        """
        ...

    @abstractmethod
    def rebuild_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        用新的文档块重建索引（删除旧索引后重新构建）。

        Args:
            chunks: 文档块列表，每项含 page_content 和 metadata
        """
        ...

    @abstractmethod
    def delete_index(self) -> None:
        """删除当前索引（包括磁盘文件）。"""
        ...
