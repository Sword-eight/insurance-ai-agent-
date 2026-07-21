"""
Insurance AI Agent - 文本切分模块
使用 RecursiveCharacterTextSplitter 将文档切分为固定大小的 Chunk。
"""

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PDF_CONFIG
from utils.logger import get_logger

logger = get_logger("rag.langchain.splitter")


class DocumentSplitter:
    """
    文档切分器。
    使用递归字符切分策略，优先在段落、句子边界处切分。
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Args:
            chunk_size: 每个 Chunk 的最大字符数
            chunk_overlap: 相邻 Chunk 的重叠字符数
        """
        self.chunk_size: int = chunk_size or PDF_CONFIG["chunk_size"]
        self.chunk_overlap: int = chunk_overlap or PDF_CONFIG["chunk_overlap"]

        # 中文字符分隔符优先
        self._separators: List[str] = [
            "\n\n",     # 段落
            "\n",       # 换行
            "。",       # 句号
            "；",       # 分号
            "，",       # 逗号
            " ",        # 空格
            "",         # 字符级
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将文档列表切分为 Chunk。

        Args:
            documents: 原始文档列表（来自 PDFLoader）

        Returns:
            切分后的 Chunk 列表，保留原始元数据并添加 Chunk 编号
        """
        if not documents:
            logger.warning("文档列表为空，无法切分")
            return []

        # 使用 LangChain 的 create_documents 进行切分
        from langchain_core.documents import Document as LCDocument

        langchain_docs: List[LCDocument] = [
            LCDocument(page_content=d["page_content"], metadata=d["metadata"])
            for d in documents
        ]

        split_docs: List[LCDocument] = self._splitter.split_documents(
            langchain_docs
        )

        # 为每个 Chunk 添加编号
        chunks: List[Dict[str, Any]] = []
        source_counter: Dict[str, int] = {}

        for doc in split_docs:
            source: str = doc.metadata.get("source", "unknown")
            if source not in source_counter:
                source_counter[source] = 0
            source_counter[source] += 1

            chunks.append({
                "page_content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "chunk_id": source_counter[source],
                    "chunk_size": len(doc.page_content),
                },
            })

        logger.info(
            f"文档切分完成: {len(documents)} 页 → {len(chunks)} 个 Chunk "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
