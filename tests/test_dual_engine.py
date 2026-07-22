"""
双引擎切换 & 接口一致性测试
验证 LangChain ↔ LlamaIndex 热切换、KnowledgeService 和 InsuranceRAGTool 行为。

所有对象通过 DI 创建（Builder → KnowledgeService，Retriever → Tool）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.base_retriever import BaseRetriever
from rag.base_index_builder import BaseIndexBuilder

# ============================================================
# 测试1：KnowledgeService LangChain 模式
# ============================================================
os.environ["RAG_ENGINE"] = "langchain"

from rag.langchain.vector_store import VectorStoreManager
from rag.langchain.retriever import LangChainRetriever
from rag.langchain.index_builder import LangChainIndexBuilder
from services.knowledge_service import KnowledgeService

vector_store = VectorStoreManager()
retriever_lc = LangChainRetriever(vector_store_manager=vector_store)
builder_lc = LangChainIndexBuilder(vector_store=vector_store)

ks_lc = KnowledgeService(builder=builder_lc, retriever=retriever_lc)
r_lc = ks_lc.get_retriever()
assert isinstance(r_lc, BaseRetriever)
assert r_lc is retriever_lc, "get_retriever() 应返回注入的同一个实例"
assert "LangChain" in type(r_lc).__name__
print("✅ KnowledgeService LangChain 模式（DI：Builder + Retriever）")

# Builder 接口一致性
assert isinstance(builder_lc, BaseIndexBuilder)
print("✅ LangChainIndexBuilder 实现了 BaseIndexBuilder")

# stats 包含所有字段
stats = ks_lc.get_stats()
for key in ["index_loaded", "index_exists", "embedding_model", "rag_engine",
            "pdf_count", "chunk_size", "chunk_overlap"]:
    assert key in stats, f"stats 缺少字段: {key}"
assert stats["rag_engine"] == "langchain"
print("✅ LangChain stats 字段完整")

# ============================================================
# 测试2：KnowledgeService LlamaIndex 模式
# ============================================================
os.environ["RAG_ENGINE"] = "llamaindex"

from rag.llamaindex.index_builder import LlamaIndexBuilder
from rag.llamaindex.retriever import LlamaIndexRetriever
from services.knowledge_service import KnowledgeService as KS2

builder_li = LlamaIndexBuilder()
retriever_li = LlamaIndexRetriever(builder=builder_li)

ks_li = KS2(builder=builder_li, retriever=retriever_li)
r_li = ks_li.get_retriever()
assert isinstance(r_li, BaseRetriever)
assert r_li is retriever_li, "get_retriever() 应返回注入的同一个实例"
assert "LlamaIndex" in type(r_li).__name__
print("✅ KnowledgeService LlamaIndex 模式（DI：Builder + Retriever）")

assert isinstance(builder_li, BaseIndexBuilder)
print("✅ LlamaIndexBuilder 实现了 BaseIndexBuilder")

stats2 = ks_li.get_stats()
assert stats2["rag_engine"] == "llamaindex"
print("✅ LlamaIndex stats 字段完整")

# ============================================================
# 测试3：InsuranceRAGTool — Retriever 注入
# ============================================================
from tools.insurance_rag_tool import InsuranceRAGTool

# LangChain Tool（注入已有的 retriever）
tool_lc = InsuranceRAGTool(retriever=retriever_lc)
assert "LangChain" in type(tool_lc._retriever).__name__
assert tool_lc._retriever is retriever_lc, "Tool 应使用注入的 retriever"
print("✅ InsuranceRAGTool LangChain 模式（DI）")

# LlamaIndex Tool（注入已有的 retriever）
tool_li = InsuranceRAGTool(retriever=retriever_li)
assert "LlamaIndex" in type(tool_li._retriever).__name__
assert tool_li._retriever is retriever_li, "Tool 应使用注入的 retriever"
print("✅ InsuranceRAGTool LlamaIndex 模式（DI）")

assert tool_lc.name == "insurance_rag_search"
assert tool_li.name == "insurance_rag_search"
print("✅ Tool name 一致")

# ============================================================
# 测试4：get_rag_engine() 兜底：非法值退回 langchain
# ============================================================
os.environ["RAG_ENGINE"] = "unsupported"
from config import get_rag_engine
assert get_rag_engine() == "langchain", f"非法引擎应兜底为 langchain"
print("✅ 非法引擎值兜底为 langchain")

# ============================================================
# 测试5：两个 Retriever 返回格式一致
# ============================================================
from rag.langchain.retriever import LangChainRetriever
from rag.llamaindex.retriever import LlamaIndexRetriever

lc_vs = VectorStoreManager()
lc = LangChainRetriever(vector_store_manager=lc_vs)
li_builder = LlamaIndexBuilder()
li = LlamaIndexRetriever(builder=li_builder)

no_index_lc = lc.retrieve("test")
no_index_li = li.retrieve("test")

required_keys = {"query", "results", "total_results"}
for key in required_keys:
    assert key in no_index_lc, f"LangChain 返回缺少: {key}"
    assert key in no_index_li, f"LlamaIndex 返回缺少: {key}"
print("✅ 无索引时返回 key 一致")

assert lc.format_retrieval_for_llm({"results": []}) == "根据当前知识库无法确定。"
assert li.format_retrieval_for_llm({"results": []}) == "根据当前知识库无法确定。"
print("✅ 空结果格式一致")

# ============================================================
# 测试6：get_rag_engine() 动态读取
# ============================================================
from config import get_rag_engine
os.environ["RAG_ENGINE"] = "langchain"
assert get_rag_engine() == "langchain"
os.environ["RAG_ENGINE"] = "llamaindex"
assert get_rag_engine() == "llamaindex"
os.environ["RAG_ENGINE"] = "garbage"
assert get_rag_engine() == "langchain"
print("✅ get_rag_engine() 动态读取 & 兜底正确")

os.environ["RAG_ENGINE"] = "langchain"

print("\n🎉 全部双引擎切换测试通过！")
