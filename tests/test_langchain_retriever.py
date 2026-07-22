"""
LangChain Retriever 测试用例
需要先有已构建的 LangChain FAISS 索引（data/vectorstore/index.faiss）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RAG_ENGINE"] = "langchain"

from rag.langchain.retriever import LangChainRetriever
from rag.langchain.vector_store import VectorStoreManager

vs = VectorStoreManager()
r = LangChainRetriever(vector_store_manager=vs)

# 加载已有索引
assert r._vector_store_manager.index_exists(), "索引不存在，请先上传PDF构建索引"
r._vector_store_manager.load_index()
print("✅ 索引加载成功")

# 测试1：基本检索
result = r.retrieve("等待期是多长时间", top_k=3)
assert result["total_results"] > 0, "应返回至少1条结果"
assert len(result["results"]) <= 3
print(f"✅ 基本检索: {result['total_results']} 条结果, 耗时 {result['retrieval_time']:.4f}s")

# 测试2：相似度在0-1之间
for res in result["results"]:
    assert 0 <= res["similarity_score"] <= 1, f"相似度 {res['similarity_score']} 不在[0,1]范围"
print(f"✅ 相似度范围正常 (top: {result['results'][0]['similarity_score']:.4f})")

# 测试3：元数据完整性
for res in result["results"]:
    assert "page_content" in res
    assert "metadata" in res
    assert "similarity_score" in res
    assert "source" in res["metadata"]
print("✅ 元数据字段完整")

# 测试4：空结果格式化
empty = {"results": []}
formatted = r.format_retrieval_for_llm(empty)
assert formatted == "根据当前知识库无法确定。"
print("✅ 空结果格式化正确")

# 测试5：正常格式化包含来源
formatted = r.format_retrieval_for_llm(result)
assert "【来源" in formatted
assert "文档:" in formatted
assert "相似度:" in formatted
print("✅ 正常格式化包含来源信息")

# 测试6：top_k 限制
r5 = r.retrieve("重大疾病", top_k=5)
assert len(r5["results"]) <= 5
r2 = r.retrieve("重大疾病", top_k=2)
assert len(r2["results"]) <= 2
print(f"✅ top_k 限制正确 (top_k=5→{len(r5['results'])}, top_k=2→{len(r2['results'])})")

# 测试7：get_stats
stats = r.get_stats()
assert "index_loaded" in stats
assert "total_vectors" in stats
print(f"✅ stats: {stats['total_vectors']} 条向量")

print("\n🎉 全部 LangChain Retriever 测试通过！")
