"""
Insurance AI Agent - 知识库服务模块
对外提供知识库的构建、重建、删除、加载、统计、文件列表六个操作。

设计原则（SRP）：
    KnowledgeService 只负责"知识管理"的门面编排，
    所有引擎细节委托给 BaseIndexBuilder 实现，
    不包含任何 if/elif 引擎分支。
"""

from typing import Dict, Any, List

from rag.base_index_builder import BaseIndexBuilder
from rag.base_retriever import BaseRetriever
from utils.logger import get_logger

logger = get_logger("services.knowledge")


class KnowledgeService:
    """
    知识库管理服务。

    对外接口：
      - build_knowledge_base()   构建索引
      - load_existing_index()    加载已有索引
      - rebuild()                重建索引
      - delete_knowledge_base()  删除索引
      - get_stats()              统计信息
      - get_pdf_files()          文档文件列表

    所有操作委托给 BaseIndexBuilder，不包含任何引擎分支。
    """

    def __init__(
        self,
        builder: BaseIndexBuilder,
        retriever: BaseRetriever,
    ) -> None:
        """
        Args:
            builder:   索引构建器（LangChainIndexBuilder 或 LlamaIndexBuilder）
            retriever: 检索器实例（用于外部获取引用）
        """
        self._builder = builder
        self._retriever = retriever
        logger.info(
            f"KnowledgeService 初始化完成: "
            f"builder={type(builder).__name__}, "
            f"retriever={type(retriever).__name__}"
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def get_retriever(self) -> BaseRetriever:
        """获取当前检索器实例。"""
        return self._retriever

    # ------------------------------------------------------------------
    # 知识库生命周期（全部委托给 BaseIndexBuilder）
    # ------------------------------------------------------------------

    def build_knowledge_base(self) -> Dict[str, Any]:
        """构建知识库索引。"""
        return self._builder.build()

    def load_existing_index(self) -> bool:
        """加载已存在的本地索引。"""
        return self._builder.load()

    def rebuild(self) -> Dict[str, Any]:
        """重建知识库——删除旧索引后重新构建。"""
        logger.info("开始重建知识库...")
        self._builder.delete()
        return self._builder.build()

    def delete_knowledge_base(self) -> Dict[str, Any]:
        """删除知识库索引。"""
        self._builder.delete()
        return {"success": True, "message": "知识库索引已删除"}

    # ------------------------------------------------------------------
    # 查询（全部委托给 BaseIndexBuilder）
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息。"""
        return self._builder.get_stats()

    def get_pdf_files(self) -> List[Dict[str, Any]]:
        """获取文档文件列表。"""
        return self._builder.list_documents()
