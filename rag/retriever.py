"""
向后兼容 stub — Retriever 现为 LangChainRetriever。
新代码请直接使用 rag.langchain.retriever.LangChainRetriever 或
通过 KnowledgeService.get_retriever() 获取 BaseRetriever 实例。
"""
from rag.langchain.retriever import LangChainRetriever as Retriever

__all__ = ["Retriever"]
