# 🛡️ Insurance AI Agent

基于 LangGraph + RAG + LoRA 的企业级保险智能助手。

## 功能

- **保险条款问答** — RAG 检索 + LLM 回答（LangChain / LlamaIndex 双引擎热切换）
- **保费估算** — 基于年龄/性别/保额/保险期限/职业类别计算预估保费
- **多轮对话** — LangGraph Agent 循环（思考→调工具→再思考）+ Checkpoint 记忆管理
- **LoRA 微调** — 保险领域 QA 自动生成 → LoRA 训练 → 模型评估
- **开发者面板** — 检索来源追溯、Agent 决策记录、引擎状态实时可见

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env              # 编辑 .env，填入 DEEPSEEK_API_KEY
streamlit run app.py              # 打开 http://localhost:8501
```

## 架构

```
                  app.py (68行组装入口)
                 /                  \
        ui/ (表示层)          application/ (应用层)
    sidebar / chat /         bootstrap / session
      components              / handlers
               │                    │
               └────────┬───────────┘
                        │
              graph/ (Agent编排层)
           AgentGraphBuilder
           agent ↔ router ↔ tools
                        │
           ┌────────────┼────────────┐
           │            │            │
    tools/ (工具层)  services/ (业务层)
  InsuranceRAGTool  RetrievalService
  PremiumCalcTool   KnowledgeService
           │        PremiumService
           │            │
           └─────┬──────┘
                 │
           rag/ (数据访问层 — Repository)
       BaseRetriever   BaseIndexBuilder
      langchain/        llamaindex/
      实现               实现
                 │
           data/ (持久化层)
        pdf/   vectorstore/
```

**调用方向：永远从上到下，不可反向。每层只依赖下一层的抽象。**

## 项目结构

```
├── app.py                          # 组装入口 (68行)
├── config.py                       # 全局配置中心
│
├── application/                    # 应用层 (纯Python, 不依赖UI)
│   ├── bootstrap.py                #   对象图工厂 — 全部DI在此组装
│   ├── session.py                  #   多会话管理
│   └── handlers.py                 #   事件处理 (chat/upload/rebuild)
│
├── ui/                             # 表示层 (Streamlit专属)
│   ├── sidebar.py                  #   侧边栏渲染
│   ├── chat.py                     #   聊天窗口渲染
│   └── components.py               #   可复用UI组件
│
├── graph/                          # Agent编排层
│   ├── graph_builder.py            #   AgentGraphBuilder (全部DI)
│   ├── state.py                    #   AgentState定义
│   ├── router.py                   #   条件路由 (tools or end)
│   └── nodes.py                    #   消息安全截断
│
├── tools/                          # 工具层 (LangChain Tool接口)
│   ├── insurance_rag_tool.py       #   知识库检索 → 委托RetrievalService
│   └── premium_calculator_tool.py  #   保费估算 → 委托PremiumService
│
├── services/                       # 业务层 (纯Python, 框架无关)
│   ├── retrieval_service.py        #   唯一检索入口 (search + format_for_llm)
│   ├── knowledge_service.py        #   索引管理门面 (build/rebuild/delete/stats)
│   └── premium_service.py          #   保费计算 (纯函数, 零依赖)
│
├── rag/                            # 数据访问层 — Repository
│   ├── base_retriever.py           #   检索抽象 + RetrievalDocument/Result
│   ├── base_index_builder.py       #   索引构建抽象
│   ├── embedding.py                #   共享 BGE Embedding (单例)
│   ├── langchain/                  #   LangChain实现
│   │   ├── retriever.py            #     FAISS检索 (归一化metadata)
│   │   ├── index_builder.py        #     构建流水线 (Loader→Splitter→FAISS)
│   │   ├── vector_store.py         #     FAISS底层封装
│   │   ├── loader.py               #     PDF/TXT加载 (PyMuPDF)
│   │   └── splitter.py             #     文本切分 (RecursiveCharacterTextSplitter)
│   └── llamaindex/                 #   LlamaIndex实现
│       ├── retriever.py            #     检索 (归一化Node→RetrievalDocument)
│       └── index_builder.py        #     索引构建 (SimpleDirectoryReader→FAISS)
│
├── memory/                         # 会话状态
│   └── state_manager.py            #   LangGraph state查询封装
│
├── prompts/                        # 系统提示词
│   └── system_prompt.py            #   Agent行为规范 + 路由规则
│
├── utils/                          # 基础设施
│   ├── logger.py                   #   日志 (单例 + 轮转文件)
│   └── helpers.py                  #   Timer / JSON读写
│
├── finetune/                       # LoRA微调模块 (独立子系统)
│   ├── dataset/                    #   QA生成 + 数据验证
│   ├── scripts/                    #   训练 + 评估
│   └── README.md                   #   微调文档
│
├── tests/                          # 测试用例
└── data/                           # 持久化
    ├── pdf/                        #   原始文档
    └── vectorstore/                #   FAISS索引
```

## 调用链示例

用户输入 "等待期多久？"：

```
app.py → application/handlers → AgentGraphBuilder.invoke()
  → _agent_node() [LLM决策→返回tool_calls]
  → route_after_agent() ["tools"]
  → _tools_node() → InsuranceRAGTool._run()
    → RetrievalService.search() → LangChainRetriever.retrieve()
      → VectorStoreManager.similarity_search() → FAISS
    → RetrievalService.format_for_llm()
  → _agent_node() [LLM看到检索结果→生成最终回答]
  → route_after_agent() ["__end__"]
→ _parse_agent_result() → UI渲染
```

## 架构原则

| 原则 | 实现 |
|------|------|
| **依赖注入 (DI)** | 所有对象由 `bootstrap.py` 创建后注入，类不自行 new 依赖 |
| **单一职责 (SRP)** | Tool只做适配 / Service只做业务 / Repository只做数据访问 |
| **开闭原则 (OCP)** | 加新RAG引擎只需加文件+1个elif, Service/Tool/Graph零改动 |
| **依赖倒置 (DIP)** | 上层只依赖抽象 (BaseRetriever/BaseIndexBuilder), 不依赖具体实现 |
| **分层调用** | 上层调下层, 永不反向 |

## RAG 引擎切换

侧边栏 `🔧 RAG 引擎` 下拉框切换 `langchain` / `llamaindex`，无需重启。

两个引擎使用独立的索引目录 (`data/vectorstore/` vs `data/vectorstore/llamaindex/`)，互不干扰。

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 编排 | LangGraph + Tool Calling |
| LLM | DeepSeek API |
| RAG 双引擎 | LangChain (PyMuPDF+FAISS) / LlamaIndex (VectorStoreIndex+FAISS) |
| 向量库 | FAISS (本地) |
| Embedding | BAAI/bge-base-zh-v1.5 (SentenceTransformer, 768维) |
| UI | Streamlit |
| 微调 | PEFT / LoRA / Qwen2.5 |

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
