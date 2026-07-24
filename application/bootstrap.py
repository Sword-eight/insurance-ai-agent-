"""
应用启动初始化模块。
负责创建全部服务实例（对象图工厂），按生命周期图自底向上组装。

不依赖 Streamlit，可单独导入和单元测试。
"""

import os
from typing import Any, Dict

from services.knowledge_service import KnowledgeService
from services.premium_service import PremiumService
from services.retrieval_service import RetrievalService
from tools.insurance_rag_tool import InsuranceRAGTool
from tools.premium_calculator_tool import PremiumCalculatorTool
from graph.graph_builder import AgentGraphBuilder
from memory.state_manager import StateManager
from utils.logger import get_logger

logger = get_logger("app.bootstrap")


def init_services(rag_engine: str = "langchain") -> Dict[str, Any]:
    """
    初始化所有服务——按生命周期图自底向上创建。

    生命周期规则：
      ① EmbeddingManager — 进程级单例，两个引擎共享
      ② BaseIndexBuilder — 引擎决定，单实例
      ③ BaseRetriever — 单实例，注入给所有需要检索的组件
      ④ KnowledgeService — 单实例（Builder + Retriever 注入）
      ⑤ PremiumService — 单实例，纯业务逻辑
      ⑥ Tools — 单实例，Retriever/Service 注入
      ⑦ AgentGraphBuilder — 单实例，Tools 注入
      ⑧ StateManager — 单实例，Graph 注入

    Args:
        rag_engine: RAG 引擎选择（"langchain" | "llamaindex"）

    Returns:
        包含各服务实例的字典
    """
    os.environ["RAG_ENGINE"] = rag_engine

    # ── Step 0: Embedding（进程级单例）──────────────────────────
    from rag.embedding import EmbeddingManager
    EmbeddingManager()  # 单例，确保模型已加载

    # ── Step 1: Builder + Retriever（引擎决定）─────────────────
    if rag_engine == "langchain":
        from rag.langchain.vector_store import VectorStoreManager
        from rag.langchain.retriever import LangChainRetriever
        from rag.langchain.index_builder import LangChainIndexBuilder

        vector_store = VectorStoreManager()
        retriever = LangChainRetriever(vector_store_manager=vector_store)
        builder = LangChainIndexBuilder(vector_store=vector_store)
    else:  # llamaindex
        from rag.llamaindex.index_builder import LlamaIndexBuilder
        from rag.llamaindex.retriever import LlamaIndexRetriever

        builder = LlamaIndexBuilder()
        retriever = LlamaIndexRetriever(builder=builder)

    # ── Step 2: KnowledgeService（仅 Builder 注入）──────────────
    knowledge_service = KnowledgeService(builder=builder)

    # 自动加载或构建知识库
    if not knowledge_service.get_stats().get("index_exists"):
        knowledge_service.build_knowledge_base()
    else:
        knowledge_service.load_existing_index()

    # ── Step 3: RetrievalService（Retriever 注入）──────────────
    retrieval_service = RetrievalService(retriever=retriever)

    # ── Step 4: PremiumService ─────────────────────────────────
    premium_service = PremiumService()

    # ── Step 5: Tools（全部依赖注入）───────────────────────────
    rag_tool = InsuranceRAGTool(service=retrieval_service)
    premium_tool = PremiumCalculatorTool(premium_service=premium_service)

    # ── Step 6: Agent + State ──────────────────────────────────
    graph_builder = AgentGraphBuilder(tools=[rag_tool, premium_tool])
    state_manager = StateManager(graph=graph_builder.graph)

    logger.info(
        f"所有服务初始化完成: engine={rag_engine}, "
        f"retriever={type(retriever).__name__}, "
        f"tools={[rag_tool.name, premium_tool.name]}"
    )

    return {
        "knowledge_service": knowledge_service,
        "retrieval_service": retrieval_service,
        "graph_builder": graph_builder,
        "state_manager": state_manager,
    }
