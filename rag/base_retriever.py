"""
Insurance AI Agent - 检索器抽象基类 & 统一数据结构
定义检索 Repository 层的最小接口和数据契约。

设计原则：
  - BaseRetriever 只做数据访问：query → RetrievalResult
  - 不做格式化（format_for_llm → Service 层）
  - 不做索引管理（get_stats/rebuild/delete → BaseIndexBuilder）
  - RetrievalDocument 是引擎归一化后的统一文档结构
  - RetrievalResult 是检索操作的完整审计记录（支持 Benchmark / Trace / UI）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ==================================================================
# 统一数据结构
# ==================================================================

@dataclass
class RetrievalDocument:
    """
    检索到的单个文档——引擎归一化后的统一结构。

    所有 Retriever 实现（LangChain / LlamaIndex / ...）
    必须在返回前将各自的 metadata 格式归一化到此结构。
    """

    content: str
    """文档文本内容。"""

    source_name: str = ""
    """来源文件名（如 "保险条款.pdf"）。"""

    source_page: int = 0
    """来源页码（从 1 开始，0 表示无页码信息）。"""

    similarity_score: float = 0.0
    """相似度分数（0.0 ~ 1.0，越高越相关）。"""

    engine: str = ""
    """检索引擎标识（如 "langchain" / "llamaindex"）。"""

    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    """原始元数据（引擎特有字段，调试用）。"""


@dataclass
class RetrievalResult:
    """
    检索操作的完整结果——审计记录。

    包含查询信息、文档列表、耗时、引擎信息，
    支持 Benchmark 对比、Trace 排查、UI 可视化。
    """

    query: str
    """原始查询文本。"""

    documents: List[RetrievalDocument] = field(default_factory=list)
    """检索到的文档列表（按相似度降序）。"""

    total_found: int = 0
    """向量库实际匹配到的总数（可能 > len(documents)，因为有 top_k 截断）。"""

    retrieval_time_ms: float = 0.0
    """纯向量检索耗时（毫秒），不含格式化。"""

    engine: str = ""
    """检索引擎标识。"""

    applied_filters: Dict[str, Any] = field(default_factory=dict)
    """检索时应用的过滤条件（预留 metadata filter 等）。"""


# ==================================================================
# 抽象基类
# ==================================================================

class BaseRetriever(ABC):
    """
    检索器抽象基类——Repository 层接口。

    唯一职责：接收 query 字符串，返回 RetrievalResult。

    所有 RAG 实现（LangChainRetriever / LlamaIndexRetriever / ...）
    必须实现此接口。
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> RetrievalResult:
        """
        执行向量检索并返回结构化结果。

        Args:
            query:           查询文本
            top_k:           返回的最大文档数
            score_threshold: 最低相似度阈值（低于此值的文档被过滤）

        Returns:
            RetrievalResult — 包含归一化后的文档列表和检索元数据
        """
        ...
