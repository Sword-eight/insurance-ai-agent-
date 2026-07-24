"""
向后兼容 stub — Retriever 现为 LangChainRetriever。
新代码请使用 RetrievalService 作为检索入口，不要直接调 Retriever。
"""
from rag.langchain.retriever import LangChainRetriever as Retriever

__all__ = ["Retriever"]
