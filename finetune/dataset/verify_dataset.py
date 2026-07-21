"""
Insurance AI Agent - 数据集验证模块
自动检查生成数据集的完整性、去重、统计分析。

检查项:
  1. instruction / output 不能为空
  2. source.file / source.page 必须存在
  3. 自动删除重复 QA（完全相同的 instruction+output）
  4. 自动删除重复问题（相同 instruction，保留最长 output）
  5. 统计每条 PDF 生成的 QA 数量

使用:
  python -m finetune.dataset.verify_dataset
  python -m finetune.dataset.verify_dataset --input insurance_train.json --output insurance_clean.json
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FINETUNE_CONFIG
from utils.logger import get_logger
from utils.helpers import load_json, save_json

logger = get_logger("finetune.verify_dataset")


class DatasetVerifier:
    """
    数据集验证器。
    负责质量检查、去重、统计。
    """

    def __init__(self, input_path: str | None = None) -> None:
        """
        Args:
            input_path: 输入 JSON 路径，默认使用配置中的 insurance_train.json
        """
        data_dir = Path(FINETUNE_CONFIG["dataset_dir"])
        self.input_path = Path(input_path) if input_path else data_dir / "insurance_train.json"
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # 验证 & 清洗
    # ------------------------------------------------------------------

    def verify_and_clean(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行完整验证和清洗流程。

        Returns:
            (清洗后的数据, 统计信息)
        """
        logger.info(f"加载数据集: {self.input_path}")

        raw_data = load_json(str(self.input_path))
        if not isinstance(raw_data, list):
            logger.error("数据集格式错误：应为 JSON 数组")
            return [], {"error": "格式错误"}

        initial_count = len(raw_data)
        stats: Dict[str, Any] = {
            "initial_count": initial_count,
            "removed_empty": 0,
            "removed_duplicate_qa": 0,
            "removed_duplicate_instruction": 0,
            "removed_missing_source": 0,
            "final_count": 0,
        }

        # Step 1: 删除空字段
        data = self._remove_empty(raw_data, stats)
        logger.info(f"  删除空字段后: {len(data)} 条 (移除 {stats['removed_empty']} 条)")

        # Step 2: 删除缺失 source 信息的
        data = self._remove_missing_source(data, stats)
        logger.info(f"  删除缺失 source 后: {len(data)} 条 (移除 {stats['removed_missing_source']} 条)")

        # Step 3: 删除完全重复的 QA
        data = self._remove_duplicate_qa(data, stats)
        logger.info(f"  去重 QA 后: {len(data)} 条 (移除 {stats['removed_duplicate_qa']} 条)")

        # Step 4: 删除重复 instruction（保留最长 output）
        data = self._remove_duplicate_instruction(data, stats)
        logger.info(f"  去重 instruction 后: {len(data)} 条 (移除 {stats['removed_duplicate_instruction']} 条)")

        stats["final_count"] = len(data)

        # 添加数据源统计
        stats["per_pdf_stats"] = self._per_pdf_stats(data)
        stats["qa_type_distribution"] = self._qa_type_distribution(data)

        return data, stats

    # ------------------------------------------------------------------
    # 清洗方法
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_empty(data: List[Dict], stats: Dict) -> List[Dict]:
        """删除 instruction 或 output 为空的条目。"""
        result = []
        for item in data:
            inst = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            if inst and out:
                result.append(item)
            else:
                stats["removed_empty"] += 1
        return result

    @staticmethod
    def _remove_missing_source(data: List[Dict], stats: Dict) -> List[Dict]:
        """删除 source.file 或 source.page 缺失的条目。"""
        result = []
        for item in data:
            src = item.get("source", {})
            if src.get("file") and src.get("page", 0) > 0:
                result.append(item)
            else:
                stats["removed_missing_source"] += 1
        return result

    @staticmethod
    def _remove_duplicate_qa(data: List[Dict], stats: Dict) -> List[Dict]:
        """删除 instruction+output 完全相同的条目。"""
        seen: Set[str] = set()
        result = []
        for item in data:
            key = f"{item.get('instruction','')}|{item.get('output','')}"
            if key not in seen:
                seen.add(key)
                result.append(item)
            else:
                stats["removed_duplicate_qa"] += 1
        return result

    @staticmethod
    def _remove_duplicate_instruction(data: List[Dict], stats: Dict) -> List[Dict]:
        """删除相同 instruction 的条目，保留 output 最长的。"""
        best: Dict[str, Dict] = {}
        for item in data:
            inst = item.get("instruction", "").strip()
            out_len = len(item.get("output", ""))
            if inst not in best or out_len > len(best[inst].get("output", "")):
                best[inst] = item

        stats["removed_duplicate_instruction"] = len(data) - len(best)
        return list(best.values())

    # ------------------------------------------------------------------
    # 统计分析
    # ------------------------------------------------------------------

    @staticmethod
    def _per_pdf_stats(data: List[Dict]) -> Dict[str, Any]:
        """统计每份 PDF 生成的 QA 数量。"""
        pdf_counts: Dict[str, int] = {}
        for item in data:
            src = item.get("source", {})
            fname = src.get("file", "unknown")
            pdf_counts[fname] = pdf_counts.get(fname, 0) + 1

        return {
            "pdf_count": len(pdf_counts),
            "per_pdf": pdf_counts,
            "avg_per_pdf": round(sum(pdf_counts.values()) / max(len(pdf_counts), 1), 1),
            "min_per_pdf": min(pdf_counts.values()) if pdf_counts else 0,
            "max_per_pdf": max(pdf_counts.values()) if pdf_counts else 0,
        }

    @staticmethod
    def _qa_type_distribution(data: List[Dict]) -> Dict[str, int]:
        """粗略统计问题类型分布（基于关键词匹配）。"""
        keywords = {
            "等待期": ["等待期"],
            "保险责任": ["保险责任", "保障范围", "保什么"],
            "免责条款": ["免责", "不赔", "除外"],
            "赔付比例": ["赔付比例", "赔付", "赔偿", "赔多少"],
            "轻症": ["轻症"],
            "重症": ["重症", "重大疾病", "重疾"],
            "理赔流程": ["理赔流程", "理赔", "索赔", "报案"],
            "现金价值": ["现金价值"],
            "退保": ["退保", "退费"],
            "投保年龄": ["投保年龄", "年龄"],
            "缴费方式": ["缴费", "保费", "交费"],
            "健康告知": ["健康告知", "告知"],
            "豁免": ["豁免"],
            "续保": ["续保"],
            "犹豫期": ["犹豫期"],
            "理赔材料": ["理赔材料", "材料", "资料"],
        }
        dist: Dict[str, int] = {k: 0 for k in keywords}
        dist["其他"] = 0

        for item in data:
            text = item.get("instruction", "") + item.get("output", "")
            matched = False
            for category, kws in keywords.items():
                if any(kw in text for kw in kws):
                    dist[category] += 1
                    matched = True
                    break
            if not matched:
                dist["其他"] += 1

        return dist

    # ------------------------------------------------------------------
    # 拆分 Train/Test + 保存
    # ------------------------------------------------------------------

    def split_and_save(
        self,
        data: List[Dict],
        train_ratio: float = 0.8,
    ) -> None:
        """
        按比例拆分为训练集和测试集，并保存为独立 JSON 文件。

        Args:
            data: 清洗后的 QA 数据
            train_ratio: 训练集比例（默认 0.8）
        """
        import random
        random.seed(42)  # 可复现

        shuffled = data.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]

        train_path = self.data_dir / "insurance_train.json"
        test_path = self.data_dir / "insurance_test.json"

        save_json(str(train_path), train_data)
        save_json(str(test_path), test_data)

        logger.info(
            f"数据集已拆分: Train={len(train_data)} 条, Test={len(test_data)} 条 "
            f"(比例 {train_ratio:.0%}:{1-train_ratio:.0%})"
        )
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Test:  {test_path}")


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def print_report(stats: Dict[str, Any]) -> None:
    """打印验证报告。"""
    print("\n" + "=" * 60)
    print("  📊 数据集验证报告")
    print("=" * 60)
    print(f"  初始数量:     {stats.get('initial_count', 0):>6} 条")
    print(f"  空字段移除:   {stats.get('removed_empty', 0):>6} 条")
    print(f"  缺失 source:  {stats.get('removed_missing_source', 0):>6} 条")
    print(f"  重复 QA:      {stats.get('removed_duplicate_qa', 0):>6} 条")
    print(f"  重复问题:     {stats.get('removed_duplicate_instruction', 0):>6} 条")
    print(f"  {'─' * 40}")
    print(f"  最终数量:     {stats.get('final_count', 0):>6} 条")

    pdf_stats = stats.get("per_pdf_stats", {})
    print(f"\n  📄 PDF 统计:")
    print(f"    文档数:     {pdf_stats.get('pdf_count', 0)}")
    print(f"    平均/PDF:   {pdf_stats.get('avg_per_pdf', 0)} 条")
    print(f"    最少/PDF:   {pdf_stats.get('min_per_pdf', 0)} 条")
    print(f"    最多/PDF:   {pdf_stats.get('max_per_pdf', 0)} 条")
    for name, count in pdf_stats.get("per_pdf", {}).items():
        print(f"      {name}: {count} 条")

    type_dist = stats.get("qa_type_distribution", {})
    if type_dist:
        print(f"\n  📝 问题类型分布:")
        for cat, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            bar = "█" * (count // max(1, max(type_dist.values()) // 20))
            print(f"    {cat:8s}: {count:>4} 条  {bar}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="验证和清洗 QA 数据集")
    parser.add_argument("--input", type=str, help="输入 JSON 文件路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    args = parser.parse_args()

    verifier = DatasetVerifier(input_path=args.input)
    cleaned, stats = verifier.verify_and_clean()

    print_report(stats)

    # 拆分并保存
    verifier.split_and_save(cleaned, train_ratio=args.train_ratio)
    print(f"\n✅ 验证完成，清洗后 {stats['final_count']} 条数据已保存")
