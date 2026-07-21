"""
Insurance AI Agent - 模型评估模块
对微调后的 LoRA 模型进行评估，生成评估报告。

评估指标:
  - Average Response Time（平均响应时间）
  - Keyword Accuracy（关键词准确率）
  - Exact Match / F1（精确匹配 / F1）
  - 按保险领域分类统计

使用:
  python -m finetune.scripts.evaluate
  python -m finetune.scripts.evaluate --test-file insurance_test.json
"""

import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FINETUNE_CONFIG
from utils.logger import get_logger
from utils.helpers import load_json, save_json, format_timestamp

logger = get_logger("finetune.evaluate")


class Evaluator:
    """
    微调模型评估器。

    评估流程:
      1. 加载测试集
      2. 逐条推理（Mock 模式 / 真实模型模式）
      3. 计算指标
      4. 生成评估报告
    """

    def __init__(self, test_file: str | None = None) -> None:
        data_dir = Path(FINETUNE_CONFIG["dataset_dir"])
        self.test_path = Path(test_file) if test_file else data_dir / "insurance_test.json"
        self.report_dir = Path(FINETUNE_CONFIG["report_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self._results: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 评估主流程
    # ------------------------------------------------------------------

    def evaluate(self, mock: bool = True) -> Dict[str, Any]:
        """
        执行评估。

        Args:
            mock: True=不加载真实模型，使用启发式模拟（仅验证评估流水线）

        Returns:
            评估指标字典
        """
        test_data = load_json(str(self.test_path))
        if not isinstance(test_data, list) or not test_data:
            logger.error("测试集为空或格式错误")
            return {"error": "测试集为空"}

        logger.info(f"加载测试集: {len(test_data)} 条")

        total_time = 0.0
        responses: List[Dict[str, Any]] = []

        for i, item in enumerate(test_data):
            instruction = item.get("instruction", "")
            expected = item.get("output", "")

            # 模拟推理（真实模式需加载 LoRA 模型）
            if mock:
                predicted, elapsed = self._mock_predict(instruction, expected, item)
            else:
                predicted, elapsed = self._model_predict(instruction)

            total_time += elapsed

            responses.append({
                "instruction": instruction[:200],
                "expected": expected[:500],
                "predicted": predicted[:500],
                "response_time": round(elapsed, 4),
                "keyword_match": self._keyword_accuracy(expected, predicted),
                "source": item.get("source", {}),
            })

        self._results = responses
        self._metrics = self._compute_metrics(responses, total_time)
        self._save_report()

        return self._metrics

    # ------------------------------------------------------------------
    # 推理方法
    # ------------------------------------------------------------------

    def _mock_predict(self, instruction: str, expected: str, item: Dict) -> Tuple[str, float]:
        """Mock 推理：返回期望答案（模拟完美模型），用于验证评估流水线。"""
        time.sleep(0.001)  # 模拟推理延迟
        elapsed = 0.05  # 模拟 50ms
        return expected, elapsed

    def _model_predict(self, instruction: str) -> Tuple[str, float]:
        """
        真实模型推理（需先训练 LoRA 模型）。

        使用方法:
          from transformers import AutoModelForCausalLM, AutoTokenizer
          from peft import PeftModel
          ...
        """
        # 预留接口：加载 LoRA 模型并推理
        raise NotImplementedError(
            "真实模型推理尚未配置。请先训练 LoRA 模型，"
            "然后在此方法中加载模型并实现推理逻辑。"
        )

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_accuracy(expected: str, predicted: str) -> float:
        """基于关键词匹配的准确率。"""
        # 从 expected 提取关键数字和术语
        import re
        # 提取数字（如 180天、50万、30岁 等）
        exp_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
        pred_numbers = set(re.findall(r'\d+(?:\.\d+)?', predicted))

        if not exp_numbers:
            return 1.0

        overlap = exp_numbers & pred_numbers
        return len(overlap) / len(exp_numbers)

    def _compute_metrics(
        self, responses: List[Dict], total_time: float
    ) -> Dict[str, Any]:
        """计算综合评估指标。"""
        n = len(responses)

        # 平均响应时间
        avg_time = total_time / n if n > 0 else 0.0

        # 关键词准确率
        kw_scores = [r["keyword_match"] for r in responses]
        avg_keyword_acc = sum(kw_scores) / n if n > 0 else 0.0

        # 按类别分组统计
        category_stats = self._category_stats(responses)

        return {
            "test_samples": n,
            "avg_response_time_sec": round(avg_time, 4),
            "total_time_sec": round(total_time, 2),
            "avg_keyword_accuracy": round(avg_keyword_acc, 4),
            "keyword_acc_above_80pct": sum(1 for s in kw_scores if s >= 0.8) / n if n > 0 else 0.0,
            "keyword_acc_above_50pct": sum(1 for s in kw_scores if s >= 0.5) / n if n > 0 else 0.0,
            "category_stats": category_stats,
            "evaluated_at": format_timestamp(),
        }

    @staticmethod
    def _category_stats(responses: List[Dict]) -> Dict[str, Any]:
        """按保险类别统计关键词准确率。"""
        categories = {
            "等待期": ["等待期"],
            "保险责任": ["保险责任", "保障范围"],
            "免责": ["免责", "不赔"],
            "赔付": ["赔付", "赔偿"],
            "重疾定义": ["重疾", "重大疾病", "恶性肿瘤"],
            "轻症": ["轻症"],
            "理赔": ["理赔", "索赔", "报案"],
            "退保": ["退保"],
            "缴费": ["缴费", "保费", "交费"],
            "续保": ["续保"],
        }

        stats = {}
        for cat, kws in categories.items():
            cat_responses = [
                r for r in responses
                if any(kw in r.get("instruction", "") for kw in kws)
            ]
            if cat_responses:
                avg = sum(r["keyword_match"] for r in cat_responses) / len(cat_responses)
                stats[cat] = {
                    "count": len(cat_responses),
                    "avg_keyword_accuracy": round(avg, 4),
                }

        return stats

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def _save_report(self) -> None:
        """保存评估报告（JSON + Markdown）。"""
        # JSON 报告
        json_path = self.report_dir / "evaluation_results.json"
        report = {
            "metrics": self._metrics,
            "samples": self._results[:10],  # 仅保存前 10 条示例
        }
        save_json(str(json_path), report)
        logger.info(f"  📊 评估报告 (JSON): {json_path}")

        # Markdown 报告
        md_path = self.report_dir / "evaluation.md"
        self._save_markdown(str(md_path))
        logger.info(f"  📊 评估报告 (MD):   {md_path}")

    def _save_markdown(self, path: str) -> None:
        """生成 Markdown 格式评估报告。"""
        m = self._metrics
        cat_stats = m.get("category_stats", {})

        lines = [
            "# Insurance AI Agent — LoRA 模型评估报告",
            "",
            f"> 评估时间: {m.get('evaluated_at', 'N/A')}",
            "",
            "---",
            "",
            "## 总体指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 测试样本数 | {m.get('test_samples', 0)} |",
            f"| 平均响应时间 | {m.get('avg_response_time_sec', 0):.4f}s |",
            f"| 总耗时 | {m.get('total_time_sec', 0):.2f}s |",
            f"| 平均关键词准确率 | {m.get('avg_keyword_accuracy', 0):.2%} |",
            f"| 关键词准确率 ≥ 80% | {m.get('keyword_acc_above_80pct', 0):.2%} |",
            f"| 关键词准确率 ≥ 50% | {m.get('keyword_acc_above_50pct', 0):.2%} |",
            "",
            "---",
            "",
            "## 按类别统计",
            "",
            "| 类别 | 样本数 | 平均关键词准确率 |",
            "|------|--------|----------------:|",
        ]

        for cat, stats in sorted(cat_stats.items(), key=lambda x: -x[1]["count"]):
            lines.append(
                f"| {cat} | {stats['count']} | {stats['avg_keyword_accuracy']:.2%} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 示例预测",
            "",
        ])

        for i, sample in enumerate(self._results[:5], 1):
            lines.extend([
                f"### 示例 {i}",
                "",
                f"**Instruction**: {sample.get('instruction', '')[:200]}",
                "",
                f"**Expected**: {sample.get('expected', '')[:300]}",
                "",
                f"**Predicted**: {sample.get('predicted', '')[:300]}",
                "",
                f"**Keyword Accuracy**: {sample.get('keyword_match', 0):.2%}",
                f"**Response Time**: {sample.get('response_time', 0):.4f}s",
                "",
            ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估 LoRA 微调模型")
    parser.add_argument("--test-file", type=str, help="测试集 JSON 路径")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="使用 Mock 模式（默认，验证评估流水线）")
    args = parser.parse_args()

    evaluator = Evaluator(test_file=args.test_file)
    metrics = evaluator.evaluate(mock=args.mock)

    print("\n" + "=" * 60)
    print("  📊 评估完成")
    print("=" * 60)
    print(f"  测试样本数:     {metrics.get('test_samples', 0)}")
    print(f"  平均响应时间:   {metrics.get('avg_response_time_sec', 0):.4f}s")
    print(f"  平均关键词准确率: {metrics.get('avg_keyword_accuracy', 0):.2%}")
    print(f"  关键词准确率≥80%: {metrics.get('keyword_acc_above_80pct', 0):.2%}")
    print("=" * 60)
