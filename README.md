# 🛡️ Insurance AI Agent

基于 LangGraph + RAG + LoRA 的企业级保险智能助手。

## 功能

- **保险条款问答** — RAG 检索 + LLM 回答（LangChain / LlamaIndex 双引擎切换）
- **保费估算** — 基于年龄/性别/保额/缴费期限计算年缴保费
- **LoRA 微调** — 保险领域 QA 自动生成 → LoRA 训练 → 模型评估
- **多轮对话** — LangGraph Agent + Checkpoint 记忆管理
- **开发者面板** — 检索来源、Agent 决策、引擎状态实时可见

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 编辑 .env，填入 DEEPSEEK_API_KEY
streamlit run app.py           # 打开 http://localhost:8501
```

## 项目结构

```
├── app.py                     # Streamlit 主入口
├── config.py                  # 全局配置中心
├── graph/                     # LangGraph Agent (StateGraph + Router)
├── tools/                     # RAG Tool + Premium Tool
├── rag/                       # 双引擎检索
│   ├── base_retriever.py      # 抽象接口
│   ├── embedding.py           # 共享 BGE Embedding
│   ├── langchain/             # LangChain 实现 (PyMuPDF + FAISS)
│   └── llamaindex/            # LlamaIndex 实现 (SimpleDirReader + FAISS)
├── services/                  # KnowledgeService + PremiumService
├── memory/                    # 对话状态管理
├── prompts/                   # System Prompt
├── finetune/                  # LoRA 微调模块
│   ├── dataset/               # QA 生成 + 数据验证
│   ├── scripts/               # 训练 + 评估
│   └── README.md              # 微调文档
└── tests/                     # 测试用例
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent | LangGraph + Tool Calling |
| LLM | DeepSeek API / Qwen2.5 LoRA |
| RAG | LangChain + LlamaIndex (双引擎) |
| 向量库 | FAISS |
| Embedding | BAAI/bge-base-zh-v1.5 |
| UI | Streamlit |
| 微调 | PEFT / LoRA / Transformers |

## RAG 引擎切换

Sidebar `🔧 RAG 引擎` 下拉框选择 `langchain` / `llamaindex`，无需重启程序。

## LoRA 微调

```bash
# 1. 生成 QA 数据
python -m finetune.dataset.build_dataset

# 2. 验证清洗
python -m finetune.dataset.verify_dataset

# 3. LoRA 训练
python finetune/scripts/run_train.py

# 4. 模型评估
python finetune/scripts/eval_real.py
```

详见 [finetune/README.md](finetune/README.md)
