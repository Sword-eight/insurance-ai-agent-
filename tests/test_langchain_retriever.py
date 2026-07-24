"""
LangChain Retriever 测试用例
验证 LangChainRetriever.retrieve() → RetrievalResult 全流程。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RAG_ENGINE"] = "langchain"

from rag.langchain.retriever import LangChainRetriever
from rag.langchain.vector_store import VectorStoreManager
from rag.base_retriever import RetrievalResult, RetrievalDocument
from services.retrieval_service import RetrievalService

vs = VectorStoreManager()
r = LangChainRetriever(vector_store_manager=vs)

# 加载已有索引
assert r._vector_store_manager.index_exists(), "索引不存在，请先上传 PDF 构建索引"
r._vector_store_manager.load_index()
print("✅ 索引加载成功")

# 测试1：retrieve() → RetrievalResult
result = r.retrieve("等待期是多长时间", top_k=3)
assert isinstance(result, RetrievalResult)
assert result.query == "等待期是多长时间"
assert result.engine == "langchain"
assert result.total_found > 0, "应返回至少1条结果"
assert len(result.documents) <= 3
assert result.retrieval_time_ms > 0
print(f"✅ retrieve: {result.total_found} 条, {result.retrieval_time_ms}ms")

# 测试2：RetrievalDocument 字段完整性
for doc in result.documents:
    assert isinstance(doc, RetrievalDocument)
    assert doc.content
    assert doc.source_name
    assert 0 <= doc.similarity_score <= 1
    assert doc.engine == "langchain"
    assert isinstance(doc.raw_metadata, dict)
print(f"✅ RetrievalDocument 字段完整 (top similarity: {result.documents[0].similarity_score:.4f})")

# 测试3：相似度范围
for doc in result.documents:
    assert 0 <= doc.similarity_score <= 1
print("✅ 相似度范围正常")

# 测试4：top_k 限制
r5 = r.retrieve("重大疾病", top_k=5)
assert len(r5.documents) <= 5
r2 = r.retrieve("重大疾病", top_k=2)
assert len(r2.documents) <= 2
print(f"✅ top_k 限制正确 (5→{len(r5.documents)}, 2→{len(r2.documents)})")

# 测试5：Service 层格式化
svc = RetrievalService(retriever=r)
fmt = svc.format_for_llm(result)
assert "【来源" in fmt
assert "文档:" in fmt
assert "相似度:" in fmt
print("✅ Service format_for_llm 包含来源信息")

# 测试6：空结果格式化
empty_fmt = svc.format_for_llm(RetrievalResult(query="", documents=[]))
assert empty_fmt == "根据当前知识库无法确定。"
print("✅ 空结果格式化正确")

print("\n🎉 全部 LangChain Retriever 测试通过！")
