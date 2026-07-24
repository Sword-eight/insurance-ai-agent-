"""
双引擎切换 & 接口一致性测试
验证 LangChain ↔ LlamaIndex 检索、RetrievalService 和 InsuranceRAGTool 行为。

所有对象通过 DI 创建（Builder → Retriever → Service → Tool）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.base_retriever import BaseRetriever, RetrievalResult, RetrievalDocument
from rag.base_index_builder import BaseIndexBuilder

# ============================================================
# 测试1：LangChain 检索链路
# ============================================================
os.environ["RAG_ENGINE"] = "langchain"

from rag.langchain.vector_store import VectorStoreManager
from rag.langchain.retriever import LangChainRetriever
from rag.langchain.index_builder import LangChainIndexBuilder
from services.retrieval_service import RetrievalService
from services.knowledge_service import KnowledgeService

vector_store = VectorStoreManager()
retriever_lc = LangChainRetriever(vector_store_manager=vector_store)
builder_lc = LangChainIndexBuilder(vector_store=vector_store)

# KnowledgeService（只做索引管理，不持有 retriever）
ks_lc = KnowledgeService(builder=builder_lc)
assert isinstance(builder_lc, BaseIndexBuilder)
print("✅ KnowledgeService LangChain 模式（仅 Builder）")

# RetrievalService（检索入口）
svc_lc = RetrievalService(retriever=retriever_lc)
assert isinstance(retriever_lc, BaseRetriever)
print("✅ RetrievalService LangChain 模式")

# ============================================================
# 测试2：LlamaIndex 检索链路
# ============================================================
os.environ["RAG_ENGINE"] = "llamaindex"

from rag.llamaindex.index_builder import LlamaIndexBuilder
from rag.llamaindex.retriever import LlamaIndexRetriever
from services.retrieval_service import RetrievalService as RS2
from services.knowledge_service import KnowledgeService as KS2

builder_li = LlamaIndexBuilder()
retriever_li = LlamaIndexRetriever(builder=builder_li)

ks_li = KS2(builder=builder_li)
assert isinstance(builder_li, BaseIndexBuilder)
print("✅ KnowledgeService LlamaIndex 模式（仅 Builder）")

svc_li = RS2(retriever=retriever_li)
assert isinstance(retriever_li, BaseRetriever)
print("✅ RetrievalService LlamaIndex 模式")

# ============================================================
# 测试3：InsuranceRAGTool — Service 注入
# ============================================================
from tools.insurance_rag_tool import InsuranceRAGTool

tool_lc = InsuranceRAGTool(service=svc_lc)
assert tool_lc.name == "insurance_rag_search"
assert tool_lc._service is svc_lc
print("✅ InsuranceRAGTool LangChain 模式（DI: Service）")

tool_li = InsuranceRAGTool(service=svc_li)
assert tool_li.name == "insurance_rag_search"
assert tool_li._service is svc_li
print("✅ InsuranceRAGTool LlamaIndex 模式（DI: Service）")

# ============================================================
# 测试4：RetrievalResult 结构
# ============================================================
no_index_lc = svc_lc.search("test")
assert isinstance(no_index_lc, RetrievalResult)
assert no_index_lc.query == "test"
assert no_index_lc.engine == "langchain"
assert isinstance(no_index_lc.documents, list)
print("✅ RetrievalResult 结构正确（LangChain）")

no_index_li = svc_li.search("test")
assert isinstance(no_index_li, RetrievalResult)
assert no_index_li.engine == "llamaindex"
print("✅ RetrievalResult 结构正确（LlamaIndex）")

# ============================================================
# 测试5：format_for_llm 统一格式化
# ============================================================
empty_fmt = svc_lc.format_for_llm(RetrievalResult(query="", documents=[]))
assert empty_fmt == "根据当前知识库无法确定。"
print("✅ 空结果格式化一致")

# 模拟有文档的格式化
doc = RetrievalDocument(
    content="等待期为90天",
    source_name="保险条款.pdf",
    source_page=3,
    similarity_score=0.87,
    engine="langchain",
)
result_with_doc = RetrievalResult(
    query="等待期",
    documents=[doc],
    total_found=1,
    engine="langchain",
)
fmt = svc_lc.format_for_llm(result_with_doc)
assert "【来源 1】" in fmt
assert "保险条款.pdf" in fmt
assert "87.00%" in fmt
assert "等待期为90天" in fmt
print("✅ format_for_llm 包含完整信息")

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
