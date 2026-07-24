"""
LlamaIndex Retriever 测试用例
验证 LlamaIndex 索引构建、retrieve() → RetrievalResult、Service 格式化全流程。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RAG_ENGINE"] = "llamaindex"

from rag.llamaindex.retriever import LlamaIndexRetriever
from rag.llamaindex.index_builder import LlamaIndexBuilder
from rag.embedding import LlamaIndexEmbeddingAdapter
from rag.base_retriever import BaseRetriever, RetrievalResult, RetrievalDocument
from services.retrieval_service import RetrievalService
from config import PDF_CONFIG

# ============================================================
# 测试1：Embedding 适配器
# ============================================================
adapter = LlamaIndexEmbeddingAdapter()
vec = adapter._get_query_embedding("测试文本")
assert len(vec) == 768, f"向量维度应为768，实际 {len(vec)}"
print(f"✅ Embedding 适配器: 维度 {len(vec)}")

# ============================================================
# 测试2：Builder + Retriever（DI）
# ============================================================
builder = LlamaIndexBuilder()
retriever = LlamaIndexRetriever(builder=builder)
assert isinstance(retriever, BaseRetriever)
print(f"✅ LlamaIndexRetriever 实现了 BaseRetriever")

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

    builder.build_index(documents=documents)
    assert builder.is_loaded
    print("✅ 索引构建成功")

    # 测试4：retrieve() → RetrievalResult
    new_builder = LlamaIndexBuilder()
    if new_builder.index_exists():
        new_builder.load_index()
    retriever2 = LlamaIndexRetriever(builder=new_builder)

    result = retriever2.retrieve("等待期是多长时间", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.query == "等待期是多长时间"
    assert result.engine == "llamaindex"
    assert result.total_found > 0
    assert len(result.documents) <= 3
    print(f"✅ retrieve: {result.total_found} 条, {result.retrieval_time_ms}ms")

    # RetrievalDocument 字段
    for doc in result.documents:
        assert isinstance(doc, RetrievalDocument)
        assert doc.content
        assert doc.engine == "llamaindex"
        assert 0 <= doc.similarity_score <= 1
    print(f"✅ RetrievalDocument 字段完整")

    # 相似度范围
    for doc in result.documents:
        assert 0 <= doc.similarity_score <= 1
    print(f"✅ 相似度范围正常 (top: {result.documents[0].similarity_score:.4f})")

    # Service 格式化（统一入口）
    svc = RetrievalService(retriever=retriever2)
    fmt = svc.format_for_llm(result)
    assert "【来源" in fmt
    assert "llamaindex" in fmt.lower() or "[llamaindex]" in fmt.lower()
    print("✅ Service format_for_llm 包含引擎标识")

else:
    print("\n⚠️  data/pdf/ 中没有文档，跳过索引构建和检索测试")

# ============================================================
# 测试5：空结果处理
# ============================================================
svc = RetrievalService(retriever=retriever)
empty_fmt = svc.format_for_llm(RetrievalResult(query="", documents=[]))
assert empty_fmt == "根据当前知识库无法确定。"
print("✅ 空结果处理正确")

# ============================================================
# 测试6：BaseRetriever 接口一致性
# ============================================================
from rag.langchain.retriever import LangChainRetriever
assert issubclass(LlamaIndexRetriever, BaseRetriever)
assert issubclass(LangChainRetriever, BaseRetriever)
print("✅ 两个 Retriever 都实现了 BaseRetriever 接口")

print("\n🎉 全部 LlamaIndex Retriever 测试通过！")
