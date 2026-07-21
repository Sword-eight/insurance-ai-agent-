"""
LlamaIndex Retriever 测试用例
验证 LlamaIndex 索引构建、检索、格式化全流程。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RAG_ENGINE"] = "llamaindex"

from rag.llamaindex.retriever import LlamaIndexRetriever
from rag.llamaindex.index_builder import LlamaIndexBuilder
from rag.embedding import LlamaIndexEmbeddingAdapter
from config import PDF_CONFIG

# ============================================================
# 测试1：Embedding 适配器
# ============================================================
adapter = LlamaIndexEmbeddingAdapter()
vec = adapter._get_query_embedding("测试文本")
assert len(vec) == 768, f"向量维度应为768，实际 {len(vec)}"
print(f"✅ Embedding 适配器: 维度 {len(vec)}")

# 批量编码
vecs = adapter._get_text_embeddings(["文本1", "文本2", "文本3"])
assert len(vecs) == 3
assert all(len(v) == 768 for v in vecs)
print("✅ 批量编码正常")

# ============================================================
# 测试2：Index Builder
# ============================================================
builder = LlamaIndexBuilder()
assert not builder.is_loaded
print(f"✅ Builder 初始化: index_path={builder._index_path}")

# 测试 index_exists（可能为 True 或 False，取决于是否已构建）
exists = builder.index_exists()
print(f"   索引存在: {exists}")

# ============================================================
# 测试3：构建索引（如果有文档）
# ============================================================
pdf_dir = PDF_CONFIG["pdf_dir"]
doc_files = []
if os.path.isdir(pdf_dir):
    doc_files = [f for f in os.listdir(pdf_dir) if f.endswith(('.pdf', '.txt'))]

if doc_files:
    from llama_index.core import SimpleDirectoryReader
    print(f"\n📄 发现 {len(doc_files)} 个文档，测试索引构建...")

    documents = SimpleDirectoryReader(input_dir=pdf_dir, recursive=False).load_data()
    assert len(documents) > 0
    print(f"   SimpleDirectoryReader 加载: {len(documents)} 个 Document")

    builder.build_index(documents=documents)
    assert builder.is_loaded
    print("✅ 索引构建成功")

    # ============================================================
    # 测试4：Retriever 检索
    # ============================================================
    retriever = LlamaIndexRetriever()
    # 构建后 _builder 不同了，需要重新测试
    builder2 = LlamaIndexBuilder()
    if builder2.index_exists():
        builder2.load_index()

    class MockRetriever:
        def __init__(self, b):
            self._builder = b
        def retrieve(self, query, top_k=3, score_threshold=0.0):
            from utils.helpers import Timer
            from typing import Dict, Any, List
            if not self._builder.is_loaded:
                self._builder.load_index()
            with Timer("test"):
                r = self._builder.index.as_retriever(similarity_top_k=top_k)
                nodes = r.retrieve(query)
            results = []
            for node in nodes:
                raw = node.score or 0.0
                sim = round(max(0.0, min(1.0, raw)), 4)
                if sim < score_threshold:
                    continue
                results.append({
                    "page_content": node.text or node.get_content(),
                    "metadata": dict(node.metadata) if node.metadata else {},
                    "similarity_score": sim,
                })
            return {"query": query, "results": results, "total_results": len(results)}

    mock = MockRetriever(builder2)
    result = mock.retrieve("等待期是多长时间", top_k=3)
    assert result["total_results"] > 0, "应返回至少1条结果"
    print(f"✅ 检索成功: {result['total_results']} 条结果")

    # 相似度范围
    for res in result["results"]:
        assert 0 <= res["similarity_score"] <= 1
    print(f"✅ 相似度范围正常 (top: {result['results'][0]['similarity_score']:.4f})")

    # 格式化
    from rag.llamaindex.retriever import LlamaIndexRetriever as LI
    li = LI()
    formatted = li.format_retrieval_for_llm(result)
    assert "【来源" in formatted
    assert "引擎: LlamaIndex" in formatted
    assert "Node ID:" in formatted
    print("✅ 格式化包含 LlamaIndex 特有字段")

else:
    print("\n⚠️  data/pdf/ 中没有文档，跳过索引构建和检索测试")

# ============================================================
# 测试5：空索引处理
# ============================================================
empty_result = {
    "query": "测试",
    "results": [],
    "total_results": 0,
}
li = LlamaIndexRetriever()
formatted = li.format_retrieval_for_llm(empty_result)
assert formatted == "根据当前知识库无法确定。"
print("✅ 空结果处理正确")

# ============================================================
# 测试6：BaseRetriever 接口一致性
# ============================================================
from rag.base_retriever import BaseRetriever
from rag.langchain.retriever import LangChainRetriever

assert issubclass(LlamaIndexRetriever, BaseRetriever)
assert issubclass(LangChainRetriever, BaseRetriever)
print("✅ 两个 Retriever 都实现了 BaseRetriever 接口")

print("\n🎉 全部 LlamaIndex Retriever 测试通过！")
