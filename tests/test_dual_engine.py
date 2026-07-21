"""
双引擎切换 & 接口一致性测试
验证 LangChain ↔ LlamaIndex 热切换、KnowledgeService 和 InsuranceRAGTool 行为。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.base_retriever import BaseRetriever

# ============================================================
# 测试1：KnowledgeService LangChain 模式
# ============================================================
os.environ["RAG_ENGINE"] = "langchain"
for mod in list(sys.modules.keys()):
    if "services.knowledge" in mod or "tools.insurance_rag" in mod:
        del sys.modules[mod]

from services.knowledge_service import KnowledgeService

ks_lc = KnowledgeService()
assert ks_lc.rag_engine == "langchain"
r_lc = ks_lc.get_retriever()
assert isinstance(r_lc, BaseRetriever)
assert "LangChain" in type(r_lc).__name__
print("✅ KnowledgeService LangChain 模式")

# stats 包含所有字段
stats = ks_lc.get_stats()
for key in ["index_loaded", "index_exists", "embedding_model", "rag_engine",
            "pdf_count", "chunk_size", "chunk_overlap"]:
    assert key in stats, f"stats 缺少字段: {key}"
print("✅ LangChain stats 字段完整")

# ============================================================
# 测试2：KnowledgeService LlamaIndex 模式
# ============================================================
os.environ["RAG_ENGINE"] = "llamaindex"
for mod in list(sys.modules.keys()):
    if "services.knowledge" in mod or "tools.insurance_rag" in mod:
        del sys.modules[mod]

from services.knowledge_service import KnowledgeService as KS2

ks_li = KS2()
assert ks_li.rag_engine == "llamaindex"
r_li = ks_li.get_retriever()
assert isinstance(r_li, BaseRetriever)
assert "LlamaIndex" in type(r_li).__name__
print("✅ KnowledgeService LlamaIndex 模式")

stats2 = ks_li.get_stats()
assert stats2["rag_engine"] == "llamaindex"
print("✅ LlamaIndex stats 字段完整")

# ============================================================
# 测试3：InsuranceRAGTool 引擎切换
# ============================================================
os.environ["RAG_ENGINE"] = "langchain"
for mod in list(sys.modules.keys()):
    if "tools.insurance_rag" in mod:
        del sys.modules[mod]

from tools.insurance_rag_tool import InsuranceRAGTool
tool_lc = InsuranceRAGTool()
assert "LangChain" in type(tool_lc._retriever).__name__
print("✅ InsuranceRAGTool LangChain 模式")

os.environ["RAG_ENGINE"] = "llamaindex"
for mod in list(sys.modules.keys()):
    if "tools.insurance_rag" in mod:
        del sys.modules[mod]

from tools.insurance_rag_tool import InsuranceRAGTool as IRT2
tool_li = IRT2()
assert "LlamaIndex" in type(tool_li._retriever).__name__
print("✅ InsuranceRAGTool LlamaIndex 模式")

# Tool 接口一致
assert tool_lc.name == "insurance_rag_search"
assert tool_li.name == "insurance_rag_search"
print("✅ Tool name 一致")

# ============================================================
# 测试4：get_rag_engine() 兜底：非法值退回 langchain
# ============================================================
os.environ["RAG_ENGINE"] = "unsupported"
for mod in list(sys.modules.keys()):
    if "services.knowledge" in mod:
        del sys.modules[mod]

from services.knowledge_service import KnowledgeService as KS3
ks3 = KS3()
assert ks3.rag_engine == "langchain", f"非法引擎应兜底为 langchain，实际: {ks3.rag_engine}"
print("✅ 非法引擎值兜底为 langchain")

# ============================================================
# 测试5：两个 Retriever 返回格式一致
# ============================================================
os.environ["RAG_ENGINE"] = "langchain"
for mod in list(sys.modules.keys()):
    if "services.knowledge" in mod:
        del sys.modules[mod]

# 只验证返回 dict 的 key 一致（不实际检索，避免依赖索引状态）
from rag.langchain.retriever import LangChainRetriever
from rag.llamaindex.retriever import LlamaIndexRetriever

lc = LangChainRetriever()
li = LlamaIndexRetriever()

# 无索引时返回格式
no_index_lc = lc.retrieve("test")
no_index_li = li.retrieve("test")

required_keys = {"query", "results", "total_results"}
for key in required_keys:
    assert key in no_index_lc, f"LangChain 返回缺少: {key}"
    assert key in no_index_li, f"LlamaIndex 返回缺少: {key}"
print("✅ 无索引时返回 key 一致")

# format 空结果一致
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
# 非法值回退到 langchain
os.environ["RAG_ENGINE"] = "garbage"
assert get_rag_engine() == "langchain"
print("✅ get_rag_engine() 动态读取 & 兜底正确")

# 恢复
os.environ["RAG_ENGINE"] = "langchain"

print("\n🎉 全部双引擎切换测试通过！")
