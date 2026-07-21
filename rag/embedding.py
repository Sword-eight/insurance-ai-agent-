"""
Insurance AI Agent - Embedding 模块
使用本地 SentenceTransformer 模型生成文本向量，不调用在线 API。
相比 HuggingFaceEmbeddings，SentenceTransformer 对 BGE 模型兼容性更好。

提供两个 Embedding 接口：
  - SentenceTransformerEmbeddingsWrapper: 实现 LangChain Embeddings 接口
  - LlamaIndexEmbeddingAdapter:  实现 LlamaIndex BaseEmbedding 接口
两者共享同一个 SentenceTransformer 模型实例（通过 EmbeddingManager 单例）。
"""

from typing import List, Any

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_CONFIG
from utils.logger import get_logger

logger = get_logger("rag.embedding")


class SentenceTransformerEmbeddingsWrapper(Embeddings):
    """
    基于 SentenceTransformer 的 LangChain Embeddings 兼容包装器。
    直接使用 sentence-transformers 库加载模型，避免 HuggingFace pipeline 兼容性问题。
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        Args:
            model_name: SentenceTransformer 模型名称
            device: 推理设备（cpu / cuda）
        """
        logger.info(f"正在加载 Embedding 模型: {model_name} ...")
        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dim: int = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding 模型加载成功: {model_name}（维度: {self._dim}）")
        except Exception as e:
            logger.error(f"Embedding 模型加载失败: {e}")
            raise

    @property
    def model(self) -> SentenceTransformer:
        """获取 SentenceTransformer 实例。"""
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量。

        Args:
            text: 查询文本

        Returns:
            向量列表（float）
        """
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档文本转换为向量（批量处理）。

        Args:
            texts: 文档文本列表

        Returns:
            向量列表的列表
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()


class EmbeddingManager:
    """
    Embedding 管理器（单例）。
    封装 SentenceTransformerEmbeddingsWrapper，对外提供统一的 Embedding 接口。
    """

    _instance: "EmbeddingManager | None" = None

    def __new__(cls) -> "EmbeddingManager":
        """单例模式，避免重复加载模型。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """初始化 Embedding 模型（仅首次调用时加载）。"""
        if self._initialized:
            return
        self._initialized = True

        model_name: str = EMBEDDING_CONFIG["model_name"]
        device: str = EMBEDDING_CONFIG.get("model_kwargs", {}).get("device", "cpu")

        self._wrapper = SentenceTransformerEmbeddingsWrapper(
            model_name=model_name,
            device=device,
        )

    @property
    def model(self) -> SentenceTransformerEmbeddingsWrapper:
        """获取兼容 LangChain Embeddings 接口的包装器。"""
        return self._wrapper

    @property
    def model_name(self) -> str:
        """获取模型名称。"""
        return EMBEDDING_CONFIG["model_name"]

    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量。

        Args:
            text: 查询文本

        Returns:
            向量列表
        """
        return self._wrapper.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档文本转换为向量。

        Args:
            texts: 文档文本列表

        Returns:
            向量列表的列表
        """
        return self._wrapper.embed_documents(texts)


# ---------------------------------------------------------------------------
# LlamaIndex Embedding 适配器
# 将现有 SentenceTransformer 包装为 LlamaIndex BaseEmbedding 接口，
# 确保 LlamaIndex 与 LangChain 共用同一套 Embedding 实现。
# ---------------------------------------------------------------------------

try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.base.embeddings.base import Embedding as LIEmbedding

    class LlamaIndexEmbeddingAdapter(BaseEmbedding):
        """
        LlamaIndex Embedding 适配器。

        将项目现有的 SentenceTransformer（通过 EmbeddingManager 单例）
        包装为 LlamaIndex 兼容的 BaseEmbedding 接口。

        注意：
          - 不加载新的模型，直接复用 EmbeddingManager 单例。
          - 不支持异步（CPU 推理），异步方法同步调用。
        """

        _embedding_manager: "EmbeddingManager" = None  # type: ignore[assignment]

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            if self._embedding_manager is None:
                # 复用全局 EmbeddingManager 单例，不重复加载模型
                LlamaIndexEmbeddingAdapter._embedding_manager = EmbeddingManager()

        @classmethod
        def class_name(cls) -> str:
            return "SentenceTransformer_Adapter"

        def _get_query_embedding(self, query: str) -> List[float]:
            """将查询文本转换为向量。"""
            return self._embedding_manager.embed_query(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            """将单个文本转换为向量。"""
            return self._embedding_manager.embed_query(text)

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            """批量将文本转换为向量。"""
            return self._embedding_manager.embed_documents(texts)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            """异步查询向量（本项目使用 CPU 推理，同步调用）。"""
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            """异步文本向量（本项目使用 CPU 推理，同步调用）。"""
            return self._get_text_embedding(text)

except ImportError:
    # LlamaIndex 未安装时，占位类避免 import 报错
    class LlamaIndexEmbeddingAdapter:  # type: ignore[no-redef]
        """Stub: LlamaIndex 未安装。"""

        def __init__(self, **kwargs: Any) -> None:
            raise ImportError(
                "LlamaIndex 未安装，无法使用 LlamaIndexEmbeddingAdapter。"
                "请执行: pip install llama-index"
            )
