"""
Insurance AI Agent - 数据集构建模块
读取 data/pdf/ 下所有保险 PDF，通过 DeepSeek API 自动生成 Alpaca 格式的 Instruction 数据。

流程:
  1. 扫描所有 PDF → 逐页提取文本
  2. 分批次调用 DeepSeek API 生成 QA
  3. 校验答案来源（确保来自 PDF 原文）
  4. 输出 Alpaca 格式 JSON

使用:
  python -m finetune.dataset.build_dataset
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from config import PDF_CONFIG, LLM_CONFIG, FINETUNE_CONFIG
from utils.logger import get_logger
from utils.helpers import Timer, save_json, load_json, format_timestamp

logger = get_logger("finetune.build_dataset")

# ────────────────────────────────────────────────────────────
# QA 生成 Prompt（中文保险领域）
# ────────────────────────────────────────────────────────────

QA_GENERATION_SYSTEM = """你是一个保险领域的资深培训专家。你的任务是根据提供的保险条款 PDF 内容，生成高质量的 Instruction 问答对，用于微调保险 AI 助手模型。

## 严格规则

1. **答案必须来源于原文** — 所有答案必须能在提供的文档中找到依据，绝对禁止编造。
2. **问题多样化** — 覆盖以下类别（尽量均匀分布）：
   - 等待期、保险责任、免责条款、赔付比例
   - 轻症定义、重症定义、理赔流程、现金价值
   - 退保规则、投保年龄、缴费方式、健康告知
   - 豁免责任、续保规则、理赔材料、犹豫期
3. **问题形式多样** — 包括：
   - 直接提问："等待期是多少天？"
   - 情景提问："如果我购买后第90天确诊重疾，能理赔吗？"
   - 对比提问："轻症和重症的赔付比例有什么不同？"
   - 条件提问："如果中途断缴，保单会怎样？"
4. **指令清晰** — instruction 字段用自然语言描述用户意图。
5. **输入可空** — 如果不需要额外上下文，input 可以为空字符串。

## 输出格式（JSON 数组）

只返回 JSON 数组，不要其他文字：

[
  {
    "instruction": "用户问：等待期是多长时间？",
    "input": "",
    "output": "根据保险条款第X条，本保险合同的等待期为180天。等待期指自保险合同生效之日起至保险公司开始承担保险责任的期间。",
    "source": {"file": "xxx.pdf", "page": 3}
  }
]

每条 output 必须包含具体数字、条款引用或原文依据。"""

QA_GENERATION_USER = """请根据以下保险条款内容，生成 {count} 条 Instruction 问答对。

要求覆盖多个维度的保险问题（等待期/保险责任/免责条款/赔付比例/轻症/重症/理赔流程/现金价值/退保/投保年龄/缴费方式/健康告知/豁免/续保/理赔材料/犹豫期）。

## 文档来源
文件名: {file_name}
页码: {page_num}

## 文档内容

{page_content}

## 输出

只返回 JSON 数组，严格遵循上述格式要求。"""


# ────────────────────────────────────────────────────────────
# Dataset Builder
# ────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    保险领域 QA 数据集构建器。

    读取 PDF → 分页提取文本 → 调用 DeepSeek 生成 QA → 输出 Alpaca 格式。
    """

    def __init__(self) -> None:
        """初始化构建器。"""
        self.pdf_dir = Path(PDF_CONFIG["pdf_dir"])
        self.output_dir = Path(FINETUNE_CONFIG["dataset_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DeepSeek 客户端
        self._client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )

        self._qa_data: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {
            "total_qa": 0,
            "pdfs_processed": 0,
            "pages_processed": 0,
            "started_at": format_timestamp(),
        }

    # ------------------------------------------------------------------
    # PDF 扫描与文本提取
    # ------------------------------------------------------------------

    def list_pdfs(self) -> List[Path]:
        """列出所有待处理的 PDF 和 TXT 文件。"""
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        txts = sorted(self.pdf_dir.glob("*.txt"))
        all_files = pdfs + txts
        logger.info(f"扫描到 {len(pdfs)} 个 PDF + {len(txts)} 个 TXT = {len(all_files)} 个文档")
        return all_files

    def extract_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        从单个文件提取逐页文本。

        Returns:
            [{"page_num": 1, "text": "...", "file_name": "xxx.pdf"}, ...]
        """
        pages: List[Dict[str, Any]] = []

        if file_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(file_path))
            for i in range(len(doc)):
                text = doc[i].get_text("text").strip()
                if text and len(text) > 20:  # 跳过过短页面
                    pages.append({
                        "page_num": i + 1,
                        "text": text,
                        "file_name": file_path.name,
                    })
            doc.close()
        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            # 按双换行拆分段落作为"页"
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                if len(para) > 20:
                    pages.append({
                        "page_num": i + 1,
                        "text": para,
                        "file_name": file_path.name,
                    })

        logger.info(f"提取完成: {file_path.name} → {len(pages)} 个有效页面")
        return pages

    # ------------------------------------------------------------------
    # QA 生成（调用 DeepSeek）
    # ------------------------------------------------------------------

    def _generate_qa_for_page(
        self,
        page_text: str,
        file_name: str,
        page_num: int,
        count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        对单页内容调用 DeepSeek，生成 count 条 QA。

        Args:
            page_text: 页面文本
            file_name: 来源文件名
            page_num: 页码
            count: 期望生成的 QA 数量

        Returns:
            QA 数据列表（Alpaca 格式）
        """
        # 截断过长的文本
        max_input_chars = 4000
        if len(page_text) > max_input_chars:
            page_text = page_text[:max_input_chars]

        user_prompt = QA_GENERATION_USER.format(
            count=count,
            file_name=file_name,
            page_num=page_num,
            page_content=page_text,
        )

        try:
            response = self._client.chat.completions.create(
                model=FINETUNE_CONFIG["qa_gen_model"],
                messages=[
                    {"role": "system", "content": QA_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=FINETUNE_CONFIG["qa_gen_temperature"],
                max_tokens=FINETUNE_CONFIG["qa_gen_max_tokens"],
            )

            content = response.choices[0].message.content or ""
            # 尝试解析 JSON（可能包裹在 ```json ... ``` 中）
            qa_list = self._parse_json_response(content)

            # 补充 source 信息
            for qa in qa_list:
                if "source" not in qa or not qa["source"]:
                    qa["source"] = {"file": file_name, "page": page_num}
                if "file" not in qa.get("source", {}):
                    qa["source"]["file"] = file_name
                if "page" not in qa.get("source", {}):
                    qa["source"]["page"] = page_num

            logger.info(f"  第 {page_num} 页生成 {len(qa_list)} 条 QA")
            return qa_list

        except Exception as e:
            logger.error(f"  第 {page_num} 页 QA 生成失败: {e}")
            return []

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """从 LLM 响应中解析 JSON 数组。"""
        # 尝试直接解析
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        import re
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # 尝试提取 [ ... ] 数组
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析 LLM 响应为 JSON，content 前 200 字符: {content[:200]}")
        return []

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def build(self, resume: bool = False) -> Dict[str, Any]:
        """
        执行完整的数据集构建流程。

        Args:
            resume: 是否从已有数据恢复（断点续传）

        Returns:
            构建统计信息
        """
        # 1. 如果 resume，先加载已有数据
        existing_path = self.output_dir / "insurance_train.json"
        if resume and existing_path.exists():
            self._qa_data = load_json(str(existing_path))
            if isinstance(self._qa_data, list):
                logger.info(f"从已有数据恢复: {len(self._qa_data)} 条 QA")
            else:
                self._qa_data = []

        # 2. 扫描 PDF
        files = self.list_pdfs()
        if not files:
            logger.warning("未找到任何 PDF/TXT 文件")
            return self._stats

        # 计算每页应生成的 QA 数量
        total_pages = 0
        for f in files:
            pages = self.extract_text(f)
            total_pages += len(pages)

        target_total = FINETUNE_CONFIG["total_qa_target"]
        qa_per_page = max(2, min(5, target_total // max(total_pages, 1)))
        logger.info(
            f"共 {len(files)} 个文件, {total_pages} 页, "
            f"目标 {target_total} 条 QA, 每页 ~{qa_per_page} 条"
        )

        # 3. 逐文件、逐页生成 QA
        for file_idx, file_path in enumerate(files, 1):
            logger.info(f"[{file_idx}/{len(files)}] 处理: {file_path.name}")

            # 跳过已在数据集中的文件（resume 模式）
            processed_sources = set()
            for qa in self._qa_data:
                src = qa.get("source", {})
                processed_sources.add(f"{src.get('file', '')}:{src.get('page', 0)}")

            pages = self.extract_text(file_path)
            self._stats["pdfs_processed"] += 1

            for page in pages:
                # resume 模式跳过已处理的页面
                key = f"{page['file_name']}:{page['page_num']}"
                if resume and key in processed_sources:
                    continue

                qa_list = self._generate_qa_for_page(
                    page_text=page["text"],
                    file_name=page["file_name"],
                    page_num=page["page_num"],
                    count=qa_per_page,
                )

                self._qa_data.extend(qa_list)
                self._stats["pages_processed"] += 1
                self._stats["total_qa"] = len(self._qa_data)

                # 每处理 10 页保存一次（防止丢失）
                if self._stats["pages_processed"] % 10 == 0:
                    self._save_intermediate()

                # API 限速：每页间隔 1 秒
                time.sleep(1)

            logger.info(f"  {file_path.name} 完成，累计 {len(self._qa_data)} 条 QA")

        # 4. 最终保存
        self._save_final()
        self._stats["finished_at"] = format_timestamp()
        self._stats["total_qa"] = len(self._qa_data)

        logger.info(f"数据集构建完成: {self._stats['total_qa']} 条 QA")
        return self._stats

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_intermediate(self) -> None:
        """保存中间结果。"""
        path = self.output_dir / "insurance_train.json"
        save_json(str(path), self._qa_data)

    def _save_final(self) -> None:
        """最终保存 + 输出统计。"""
        self._save_intermediate()

        # 保存构建统计
        stats_path = self.output_dir / "build_stats.json"
        save_json(str(stats_path), self._stats)

        logger.info(f"数据集已保存到 {self.output_dir}")
        logger.info(f"  QA 总数: {self._stats['total_qa']}")
        logger.info(f"  处理 PDF 数: {self._stats['pdfs_processed']}")
        logger.info(f"  处理页面数: {self._stats['pages_processed']}")


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="构建保险领域 QA 数据集")
    parser.add_argument("--resume", action="store_true", help="从已有数据恢复（断点续传）")
    args = parser.parse_args()

    builder = DatasetBuilder()
    with Timer("数据集构建"):
        stats = builder.build(resume=args.resume)

    print(f"\n✅ 数据集构建完成: {stats['total_qa']} 条 QA")
