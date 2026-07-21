# Insurance AI Agent — LoRA 微调模块

## 为什么需要 LoRA？

当前项目通过 RAG（检索增强生成）实现保险知识问答。RAG 的优点是知识可随时更新，不需要重新训练模型。但 RAG 也有局限：

| 场景 | RAG | LoRA 微调 |
|------|-----|----------|
| 保险条款更新 | ✅ 直接更新知识库 | ❌ 需重新训练 |
| 回答风格一致性 | ⚠️ 依赖 Prompt 约束 | ✅ 模型内化 |
| 推理速度 | 较慢（检索+LLM） | 快（直接生成） |
| 领域术语理解 | ⚠️ 通用模型 | ✅ 领域适应 |
| 离线场景 | ❌ 需要 LLM API | ✅ 本地部署 |

**LoRA + RAG 互补**：LoRA 让模型深度理解保险术语和回答风格，RAG 保证知识的时效性。

---

## LoRA 与 RAG 的区别

```
RAG:
  用户提问 → 检索相关条款 → LLM(通用) + Context → 回答

LoRA + RAG:
  用户提问 → 检索相关条款 → LLM(保险微调) + Context → 回答
                              ↑
                      LoRA Adapter
                   (保险领域知识内化)
```

LoRA 微调后，DeepSeek 模型会更准确地：
- 使用保险术语（等待期/豁免/现金价值）
- 保持统一回答风格
- 减少幻觉（因为见过大量保险 QA）

---

## 目录结构

```
finetune/
├── README.md                      ← 本文档
├── SYSTEM_DESIGN.md               ← 系统设计文档
├── dataset/
│   ├── build_dataset.py           ← QA 数据集构建（DeepSeek API 生成）
│   └── verify_dataset.py          ← 数据集验证 & 清洗 & 统计
├── scripts/
│   ├── train_lora.py              ← 生成 LLaMA-Factory 训练配置
│   ├── train_lora.sh              ← 训练脚本（由 train_lora.py 生成）
│   ├── llamafactory_dataset_info.json ← LLaMA-Factory 数据集注册
│   └── evaluate.py                ← 模型评估 & 报告
├── data/
│   ├── insurance_train.json       ← 训练集（80%）
│   ├── insurance_test.json        ← 测试集（20%）
│   └── build_stats.json           ← 构建统计
├── outputs/                       ← LLaMA-Factory 训练输出
│   └── lora_train_config.json     ← 训练配置参考
└── reports/
    ├── evaluation_results.json    ← 评估结果（JSON）
    └── evaluation.md              ← 评估报告（Markdown）
```

---

## 数据集构建流程

```
data/pdf/ 下所有保险 PDF
    ↓
build_dataset.py  逐页读取 → DeepSeek API 生成 QA
    ↓
insurance_train.json  (Alpaca 格式, 300~500 条)
    ↓
verify_dataset.py  去重 → 清洗 → 统计 → 拆分
    ↓
insurance_train.json (80%)  +  insurance_test.json (20%)
```

### Step 1: 构建数据集

```bash
# 生成 QA 数据集（使用 DeepSeek API）
python -m finetune.dataset.build_dataset

# 如果中途中断，可断点续传
python -m finetune.dataset.build_dataset --resume
```

每条 QA 格式：

```json
{
  "instruction": "用户问：等待期是多长时间？",
  "input": "",
  "output": "根据保险条款第四条，本保险合同的等待期为180天...",
  "source": {
    "file": "中国人寿重大疾病保险条款.txt",
    "page": 3
  }
}
```

### Step 2: 验证和清洗

```bash
# 验证数据集质量
python -m finetune.dataset.verify_dataset

# 指定输入文件和拆分比例
python -m finetune.dataset.verify_dataset \
    --input finetune/data/insurance_train.json \
    --train-ratio 0.8
```

输出报告包含：
- QA 总数 / PDF 分布 / 问题类型分布
- 删除了多少空字段、重复 QA、重复问题

---

## 如何训练

### 前置条件

```bash
# 1. 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 2. 注册数据集
# 将 finetune/scripts/llamafactory_dataset_info.json 的内容
# 合并到 LLaMA-Factory/data/dataset_info.json

# 3. 复制数据
cp finetune/data/insurance_train.json LLaMA-Factory/data/
cp finetune/data/insurance_test.json  LLaMA-Factory/data/
```

### 生成训练配置

```bash
python -m finetune.scripts.train_lora
```

生成的文件：
- `scripts/llamafactory_dataset_info.json` — 数据集注册信息
- `scripts/train_lora.sh` — 一键训练脚本
- `outputs/lora_train_config.json` — 完整参数参考

### 默认训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | Qwen2.5-1.5B-Instruct | 1.5B 参数，中文友好 |
| LoRA Rank | 16 | 低秩矩阵秩 |
| LoRA Alpha | 32 | 缩放因子 |
| Epochs | 3 | 训练轮数 |
| Learning Rate | 5e-5 | 学习率 |
| Batch Size | 4 × 4 (grad accum) | 有效批次 16 |
| Max Length | 2048 | 最大序列长度 |

### 开始训练

```bash
bash finetune/scripts/train_lora.sh
```

训练完成后，LoRA Adapter 在 `LLaMA-Factory/output/insurance_lora/`。

---

## 如何评估

```bash
# Mock 模式（验证评估流水线）
python -m finetune.scripts.evaluate

# 真实模型评估（需先训练完成）
python -m finetune.scripts.evaluate --test-file finetune/data/insurance_test.json
```

评估指标：

| 指标 | 说明 |
|------|------|
| Average Response Time | 平均推理时间 |
| Keyword Accuracy | 关键词（数字/术语）匹配率 |
| Exact Match (F1) | 精确匹配度 |
| 分类统计 | 按保险类别（等待期/理赔/重疾...）分别统计 |

评估报告保存在 `reports/evaluation.md`。

---

## 如何替换 Adapter

训练完成后，新的 LoRA Adapter 可以替换当前 DeepSeek API 调用：

```python
# 方式 1: 在 graph/nodes.py 中加载本地 LoRA 模型
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(model, "outputs/insurance_lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

```python
# 方式 2: 通过 vLLM 部署 LoRA 服务（推荐生产环境）
# vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#     --enable-lora \
#     --lora-modules insurance=outputs/insurance_lora
```

---

## 持续迭代

```
新增 PDF → build_dataset.py --resume → verify → 重新训练
```

`--resume` 参数支持断点续传：已有 QA 不会被删除，仅补充新 PDF 的 QA。
