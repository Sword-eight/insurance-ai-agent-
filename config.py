"""
Insurance AI Agent - 全局配置模块
集中管理所有配置项，遵循 SOLID 原则中的单一职责原则。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------
# 项目根目录
# ------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 加载环境变量
# ------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ------------------------------------------------------------
# LLM 配置（DeepSeek）
# ------------------------------------------------------------
LLM_CONFIG: dict = {
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "temperature": 0.0,
    "max_tokens": 4096,
}

# ------------------------------------------------------------
# Embedding 配置（本地 HuggingFace）
# ------------------------------------------------------------
EMBEDDING_CONFIG: dict = {
    "model_name": "BAAI/bge-base-zh-v1.5",
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": True},
}

# ------------------------------------------------------------
# 向量数据库配置（FAISS）
# ------------------------------------------------------------
VECTOR_STORE_CONFIG: dict = {
    "index_path": str(PROJECT_ROOT / "data" / "vectorstore"),
    "top_k": 5,
}

# ------------------------------------------------------------
# PDF 文档配置
# ------------------------------------------------------------
PDF_CONFIG: dict = {
    "pdf_dir": str(PROJECT_ROOT / "data" / "pdf"),
    "chunk_size": 500,
    "chunk_overlap": 100,
}

# ------------------------------------------------------------
# Memory 配置
# ------------------------------------------------------------
MEMORY_CONFIG: dict = {
    "max_context_rounds": 5,
}

# ------------------------------------------------------------
# RAG 引擎选择: "langchain" | "llamaindex"
# 可通过环境变量 RAG_ENGINE 覆盖。
# 注意：使用 get_rag_engine() 函数动态读取以支持运行时热切换，
#       不要直接使用模块级 RAG_ENGINE 常量（仅在启动时求值一次）。
# ------------------------------------------------------------
def get_rag_engine() -> str:
    """动态获取 RAG 引擎配置（支持运行时热切换）。"""
    engine: str = os.getenv("RAG_ENGINE", "langchain")
    if engine not in ("langchain", "llamaindex"):
        engine = "langchain"
    return engine


# 启动时的默认值（向后兼容旧写法，新代码请用 get_rag_engine()）
RAG_ENGINE: str = get_rag_engine()

# ------------------------------------------------------------
# LoRA 微调配置
# ------------------------------------------------------------
FINETUNE_CONFIG: dict = {
    # 数据集
    "dataset_dir": str(PROJECT_ROOT / "finetune" / "data"),
    "qa_per_pdf_min": 30,
    "qa_per_pdf_max": 50,
    "total_qa_target": 500,
    "train_ratio": 0.8,
    # LLaMA-Factory 训练参数
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    "epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "max_length": 2048,
    # 输出
    "output_dir": str(PROJECT_ROOT / "finetune" / "outputs"),
    "report_dir": str(PROJECT_ROOT / "finetune" / "reports"),
    # QA 生成
    "qa_gen_model": "deepseek-chat",
    "qa_gen_temperature": 0.7,
    "qa_gen_max_tokens": 4096,
}
PREMIUM_RATE_PATH: str = str(PROJECT_ROOT / "config" / "premium_rate.json")

# ------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------
LOG_CONFIG: dict = {
    "log_dir": str(PROJECT_ROOT / "logs"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
}

# ------------------------------------------------------------
# Streamlit 配置
# ------------------------------------------------------------
STREAMLIT_CONFIG: dict = {
    "page_title": "Insurance AI Agent",
    "page_icon": "🛡️",
    "layout": "wide",
}
