"""
Insurance AI Agent - PDF 加载模块
使用 PyMuPDF (fitz) 解析 PDF 文档，提取文本内容和元数据。
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from config import PDF_CONFIG
from utils.logger import get_logger

logger = get_logger("rag.langchain.loader")


class PDFLoader:
    """
    PDF 文档加载器。
    负责扫描 PDF 目录、解析文档内容、提取章节等信息。
    """

    def __init__(self, pdf_dir: str | None = None) -> None:
        """
        Args:
            pdf_dir: PDF 文件目录路径，默认使用全局配置
        """
        self.pdf_dir: Path = Path(pdf_dir or PDF_CONFIG["pdf_dir"])
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def list_pdfs(self) -> List[Path]:
        """
        列出目录中所有 PDF 文件。

        Returns:
            PDF 文件路径列表
        """
        pdf_files: List[Path] = sorted(self.pdf_dir.glob("*.pdf"))
        logger.info(f"扫描到 {len(pdf_files)} 个 PDF 文件: {[f.name for f in pdf_files]}")
        return pdf_files

    def list_documents(self) -> List[Path]:
        """
        列出目录中所有可处理的文档（PDF + TXT）。

        Returns:
            文档文件路径列表
        """
        pdf_files: List[Path] = sorted(self.pdf_dir.glob("*.pdf"))
        txt_files: List[Path] = sorted(self.pdf_dir.glob("*.txt"))
        all_files: List[Path] = pdf_files + txt_files
        logger.info(
            f"扫描到 {len(pdf_files)} 个 PDF + {len(txt_files)} 个 TXT = "
            f"{len(all_files)} 个文档"
        )
        return all_files

    def load_pdf(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        加载单个 PDF 文件，提取每页文本和元数据。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            文档列表，每个元素包含文本内容和元数据
        """
        pdf_path = Path(pdf_path)
        documents: List[Dict[str, Any]] = []
        pdf_name: str = pdf_path.name

        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"加载 PDF: {pdf_name}，共 {len(doc)} 页")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text: str = page.get_text("text").strip()

                # 跳过空白页
                if not text:
                    continue

                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": pdf_name,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "file_path": str(pdf_path),
                    },
                })

            doc.close()

        except Exception as e:
            logger.error(f"加载 PDF 失败 {pdf_name}: {e}")
            raise

        logger.info(f"PDF {pdf_name} 提取完成，共 {len(documents)} 个有效页面")
        return documents

    def load_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        加载目录中所有可处理文档（PDF + TXT）。

        Returns:
            所有文档的列表
        """
        all_documents: List[Dict[str, Any]] = []
        doc_files: List[Path] = self.list_documents()

        if not doc_files:
            logger.warning(f"文档目录为空: {self.pdf_dir}")
            return all_documents

        for doc_path in doc_files:
            try:
                if doc_path.suffix.lower() == ".pdf":
                    documents: List[Dict[str, Any]] = self.load_pdf(doc_path)
                elif doc_path.suffix.lower() == ".txt":
                    documents = self.load_txt(doc_path)
                else:
                    continue
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"跳过文件 {doc_path.name}: {e}")
                continue

        logger.info(f"所有文档加载完成，共 {len(all_documents)} 个页面文档")
        return all_documents

    def load_txt(self, txt_path: str | Path) -> List[Dict[str, Any]]:
        """
        加载单个 TXT 文件，按段落拆分并生成元数据。

        Args:
            txt_path: TXT 文件路径

        Returns:
            文档列表，每个元素包含文本内容和元数据
        """
        txt_path = Path(txt_path)
        documents: List[Dict[str, Any]] = []

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text: str = f.read()

            if not full_text.strip():
                logger.warning(f"TXT 文件为空: {txt_path.name}")
                return documents

            # 按双换行拆分为段落，模拟 PDF 的"页"
            paragraphs: List[str] = full_text.split("\n\n")
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue

                documents.append({
                    "page_content": para,
                    "metadata": {
                        "source": txt_path.name,
                        "page": i + 1,
                        "total_pages": len(paragraphs),
                        "file_path": str(txt_path),
                    },
                })

            logger.info(
                f"加载 TXT: {txt_path.name}，共 {len(documents)} 个段落"
            )

        except Exception as e:
            logger.error(f"加载 TXT 失败 {txt_path.name}: {e}")
            raise

        return documents

    def get_pdf_count(self) -> int:
        """
        获取文档文件数量。

        Returns:
            文档文件数量（PDF + TXT）
        """
        return len(self.list_documents())
