# Insurance AI Agent — LoRA 微调模块系统设计

> **文档版本**: v1.0
> **设计目标**: 在不修改现有 RAG/Agent/Graph/Memory/Tool/Prompt 模块的前提下，新增 LoRA 微调管线

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────┐
│                  Insurance AI Agent                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │ Streamlit│  │ LangGraph│  │    RAG Pipeline    │ │
│  │   UI     │  │  Agent   │  │ (LangChain/Llama)  │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│       ↑              ↑                ↑              │
│       │              │                │              │
│  ┌────┴──────────────┴────────────────┴──────────┐  │
│  │              DeepSeek API (通用 LLM)            │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────── 新增: finetune/ ──────────────┐  │
│  │                                                  │
│  │  dataset/          scripts/        reports/      │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  │ build    │    │ train    │    │ eval     │   │
│  │  │ dataset  │ →  │ lora     │ →  │ report   │   │
│  │  └──────────┘    └──────────┘    └──────────┘   │
│  │       ↓               ↓               ↓         │
│  │  data/           outputs/        reports/        │
│  │  train.json      lora adapter    eval.md         │
│  │  test.json                                      │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 2. 模块职责

### 2.1 与现有系统的关系

```
现有系统（不做任何修改）:
  graph/        → LangGraph Agent 工作流
  tools/        → insurance_rag_tool / premium_calculator_tool
  rag/          → LangChain + LlamaIndex 双引擎检索
  memory/       → 对话状态管理
  prompts/      → System Prompt
  config.py     → 全局配置（新增 FINETUNE_CONFIG）
  utils/        → Logger / Timer / JSON helpers

新增模块（独立运作，零耦合）:
  finetune/
    dataset/    → 数据集构建 + 验证
    scripts/    → 训练配置生成 + 模型评估
    data/       → 训练/测试数据
    outputs/    → LoRA Adapter
    reports/    → 评估报告
```

### 2.2 数据流

```
                            data/pdf/*.pdf
                                  │
                    ┌─────────────┴─────────────┐
                    │   finetune/dataset/        │
                    │   build_dataset.py         │
                    │   ↓ 逐页提取文本            │
                    │   ↓ DeepSeek API 生成 QA   │
                    │   ↓ Alpaca 格式输出         │
                    └─────────────┬─────────────┘
                                  │
                        insurance_train.json
                                  │
                    ┌─────────────┴─────────────┐
                    │   finetune/dataset/        │
                    │   verify_dataset.py        │
                    │   ↓ 去重                   │
                    │   ↓ 清洗                   │
                    │   ↓ 统计                   │
                    │   ↓ 拆分 80/20             │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │               │            │
            insurance_train.json   insurance_test.json
                    │               │
                    │               │
    ┌───────────────┴───┐   ┌──────┴────────────┐
    │  LLaMA-Factory    │   │  evaluate.py       │
    │  LoRA 微调         │   │  模型评估           │
    │  ↓ Qwen2.5-1.5B  │   │  ↓ 关键词准确率     │
    │  ↓ LoRA Adapter   │   │  ↓ 分类统计         │
    │  ↓ 训练报告        │   │  ↓ evaluation.md   │
    └───────────────────┘   └───────────────────┘
```

---

## 3. 接口设计

### 3.1 共享依赖

所有 finetune 模块复用项目现有基础设施：

| 依赖 | 来源 |
|------|------|
| Logger | `utils.logger.get_logger("finetune.xxx")` |
| JSON 读写 | `utils.helpers.load_json / save_json` |
| Timer | `utils.helpers.Timer` |
| DeepSeek API | `openai.OpenAI`（使用 `LLM_CONFIG`） |
| PDF 配置 | `config.PDF_CONFIG`, `config.FINETUNE_CONFIG` |

### 3.2 类图

```
DatasetBuilder
  ├── list_pdfs() → List[Path]
  ├── extract_text(file) → List[Dict]
  ├── build(resume=False) → Dict
  └── _generate_qa_for_page() → List[Dict]

DatasetVerifier
  ├── verify_and_clean() → (List[Dict], Dict)
  └── split_and_save(data, ratio) → None

TrainConfigGenerator
  └── generate_all(dry_run) → Dict

Evaluator
  ├── evaluate(mock=True) → Dict
  └── _save_report() → None
```

### 3.3 数据格式

**Alpaca 格式（输入/输出）**:

```json
[
  {
    "instruction": "用户问：等待期是多长时间？",
    "input": "",
    "output": "根据保险条款第四条，本保险合同的等待期为180天...",
    "source": {
      "file": "中国人寿重大疾病保险条款.txt",
      "page": 3
    }
  }
]
```

**统计格式**:

```json
{
  "initial_count": 500,
  "removed_empty": 0,
  "removed_duplicate_qa": 5,
  "removed_duplicate_instruction": 12,
  "removed_missing_source": 0,
  "final_count": 483,
  "per_pdf_stats": {
    "pdf_count": 3,
    "per_pdf": {"a.pdf": 200, "b.pdf": 283}
  }
}
```

---

## 4. 配置管理

所有 finetune 参数集中在 `config.FINETUNE_CONFIG`：

```python
FINETUNE_CONFIG = {
    "dataset_dir":       "finetune/data/",
    "qa_per_pdf_min":    30,        # 每份 PDF 最少生成 30 条
    "qa_per_pdf_max":    50,        # 每份 PDF 最多生成 50 条
    "total_qa_target":   500,       # 全局目标 500 条
    "train_ratio":       0.8,       # 80% 训练
    "base_model":        "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_rank":         16,
    "lora_alpha":        32,
    "epochs":            3,
    "learning_rate":     5e-5,
}
```

可通过修改 `config.py` 或环境变量调整参数。

---

## 5. 如何进行 LoRA + RAG 协同推理

协同工作流（需加载 LoRA Adapter 后）：

```
用户提问
    ↓
LangGraph Agent（不变）
    ↓
agent_node 决策
    ↓
调用 insurance_rag_search Tool
    ↓
RAG 检索相关条款（不变）
    ↓
将检索结果 + 用户问题 → 送入 本地 LoRA 模型（替代 DeepSeek API）
    ↓
LoRA 模型生成保险专业回答
    ↓
返回用户
```

简单替换：将 `graph/nodes.py` 中 `ChatOpenAI` 的 `base_url` 指向本地 vLLM/LoRA 服务即可。

---

## 6. 扩展性

| 扩展场景 | 操作 |
|---------|------|
| 新增 PDF | 放入 `data/pdf/`，运行 `build_dataset.py --resume` |
| 扩充 QA | 修改 `FINETUNE_CONFIG["total_qa_target"]`，重新运行 |
| 更换基础模型 | 修改 `FINETUNE_CONFIG["base_model"]`，重新生成配置 |
| 新增第三种 RAG | 继承 `BaseRetriever`，无需修改 finetune |
| 替换评估指标 | 修改 `evaluate.py` 中的 `_compute_metrics()` |

---

## 7. 文件清单

```
finetune/
├── __init__.py                          # 包初始化
├── README.md                            # 用户文档
├── SYSTEM_DESIGN.md                     # 本文档
├── dataset/
│   ├── build_dataset.py                 # QA 生成 (450+ 行)
│   └── verify_dataset.py                # 验证清洗 (280+ 行)
├── scripts/
│   ├── train_lora.py                    # 训练配置生成 (220+ 行)
│   ├── train_lora.sh                    # 训练脚本 (生成)
│   ├── llamafactory_dataset_info.json   # 数据集注册 (生成)
│   └── evaluate.py                      # 模型评估 (280+ 行)
├── data/                                # 训练数据
│   ├── insurance_train.json             # 训练集
│   ├── insurance_test.json              # 测试集
│   └── build_stats.json                 # 构建统计
├── outputs/                             # 训练输出
└── reports/                             # 评估报告
```
