"""Real model evaluation — 加载训练好的 LoRA Adapter 对测试集进行评估。"""
import json, time, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen2.5-0.5B-Instruct\snapshots\master"
ADAPTER_PATH = str(ROOT / "finetune" / "outputs" / "adapter")
TEST_PATH = ROOT / "finetune" / "data" / "insurance_test.json"
REPORT_DIR = ROOT / "finetune" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("LoRA 模型评估")
print("=" * 50)

# ── Load Model ──
print("\n[1/4] Loading model + LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(f"  Model + LoRA loaded (GPU)")

# ── Load Test Data ──
print("\n[2/4] Loading test data...")
with open(TEST_PATH, encoding="utf-8") as f:
    test_data = json.load(f)
print(f"  {len(test_data)} samples")

# ── Run Inference ──
print("\n[3/4] Running inference...")
results = []
total_time = 0.0

for i, item in enumerate(test_data):
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    expected = item.get("output", "")
    source = item.get("source", {})

    # 构建 prompt
    if inp:
        prompt = f"<|im_start|>user\n{instruction}\n{inp}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.1,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    total_time += elapsed

    predicted = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 关键词准确率
    exp_nums = set(re.findall(r'\d+(?:\.\d+)?', expected))
    pred_nums = set(re.findall(r'\d+(?:\.\d+)?', predicted))
    kw_acc = len(exp_nums & pred_nums) / len(exp_nums) if exp_nums else 1.0

    results.append({
        "instruction": instruction[:200],
        "expected": expected[:500],
        "predicted": predicted[:500],
        "response_time": round(elapsed, 4),
        "keyword_match": round(kw_acc, 4),
        "source": source,
    })

    if (i + 1) % 3 == 0:
        print(f"  {i+1}/{len(test_data)} | kw_acc={kw_acc:.2%} | {elapsed:.2f}s")

avg_time = total_time / len(test_data)
avg_kw = sum(r["keyword_match"] for r in results) / len(results)
kw_80 = sum(1 for r in results if r["keyword_match"] >= 0.8) / len(results)

# ── Category Stats ──
categories = {
    "等待期": ["等待期"], "保险责任": ["保险责任", "保障范围"],
    "免责": ["免责", "不赔"], "赔付": ["赔付", "赔偿"],
    "重疾": ["重疾", "重大疾病", "恶性肿瘤"], "轻症": ["轻症"],
    "理赔": ["理赔", "索赔", "报案"], "退保": ["退保"],
    "缴费": ["缴费", "保费"], "续保": ["续保"],
}
cat_stats = {}
for cat, kws in categories.items():
    cat_res = [r for r in results if any(kw in r.get("instruction", "") for kw in kws)]
    if cat_res:
        cat_stats[cat] = {
            "count": len(cat_res),
            "avg_kw": round(sum(r["keyword_match"] for r in cat_res) / len(cat_res), 4),
        }

# ── Save Report ──
print("\n[4/4] Generating reports...")

# JSON
json_path = REPORT_DIR / "evaluation_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metrics": {
            "test_samples": len(test_data),
            "avg_response_time": round(avg_time, 4),
            "avg_keyword_accuracy": round(avg_kw, 4),
            "keyword_acc_above_80pct": round(kw_80, 4),
            "category_stats": cat_stats,
        },
        "samples": results[:5],
    }, f, ensure_ascii=False, indent=2)
print(f"  JSON: {json_path}")

# Markdown
md_lines = [
    "# Insurance AI Agent — LoRA 模型评估报告",
    f"\n> 模型: Qwen2.5-0.5B-Instruct + LoRA",
    f"> 测试样本: {len(test_data)} 条",
    "",
    "## 总体指标",
    "",
    "| 指标 | 值 |",
    "|------|-----|",
    f"| 测试样本数 | {len(test_data)} |",
    f"| 平均响应时间 | {avg_time:.4f}s |",
    f"| 平均关键词准确率 | {avg_kw:.2%} |",
    f"| 关键词准确率 ≥ 80% | {kw_80:.2%} |",
    "",
    "## 按类别统计",
    "",
    "| 类别 | 样本数 | 关键词准确率 |",
    "|------|--------|------------:|",
]
for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["count"]):
    md_lines.append(f"| {cat} | {s['count']} | {s['avg_kw']:.2%} |")

md_lines.extend([
    "",
    "## 预测示例",
    "",
])
for i, r in enumerate(results[:5], 1):
    md_lines.append(f"### 示例 {i} (kw_acc={r['keyword_match']:.2%}, {r['response_time']:.2f}s)")
    md_lines.append(f"\n**Instruction**: {r['instruction'][:200]}")
    md_lines.append(f"\n**Expected**: {r['expected'][:300]}")
    md_lines.append(f"\n**Predicted**: {r['predicted'][:300]}")
    md_lines.append("")

md_path = REPORT_DIR / "evaluation.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"  Markdown: {md_path}")

# ── Summary ──
print("\n" + "=" * 50)
print("  评估完成")
print("=" * 50)
print(f"  测试样本:     {len(test_data)}")
print(f"  平均响应时间: {avg_time:.4f}s")
print(f"  关键词准确率: {avg_kw:.2%}")
print(f"  准确率 ≥80%:  {kw_80:.2%}")
print(f"  报告:         {REPORT_DIR}")
print("=" * 50)
