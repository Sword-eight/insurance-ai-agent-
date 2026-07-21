"""
Insurance AI Agent - LangChain Embedding 模块
从 rag.embedding 重新导出共享的 Embedding 实现。

注意：整个项目只有一套 Embedding 实现（rag.embedding），
      此模块仅作为 LangChain 子包的一致性入口。
"""

from rag.embedding import SentenceTransformerEmbeddingsWrapper, EmbeddingManager

__all__ = ["SentenceTransformerEmbeddingsWrapper", "EmbeddingManager"]
