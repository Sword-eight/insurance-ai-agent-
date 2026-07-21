"""
Insurance AI Agent - LlamaIndex 检索器
实现 BaseRetriever 接口，使用 LlamaIndex as_retriever() 进行检索。

注意：
  - 不使用 QueryEngine.query()，不调用 LLM。
  - 只用 index.as_retriever().retrieve() 获取 Node 列表。
  - 返回格式与 LangChainRetriever 完全一致，上层调用方无感知。
"""

from typing import List, Dict, Any, Optional

from rag.base_retriever import BaseRetriever
from rag.llamaindex.index_builder import LlamaIndexBuilder
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("rag.llamaindex.retriever")


def _extract_chunk_order(node: Any) -> int:
    """从 Node relationships 推断 chunk 在文档中的顺序位置。"""
    try:
        if hasattr(node, 'relationships') and node.relationships:
            from llama_index.core.schema import NodeRelationship
            # NEXT 存在 → 不是最后一个；PREVIOUS 存在 → 不是第一个
            has_prev = NodeRelationship.PREVIOUS in node.relationships
            has_next = NodeRelationship.NEXT in node.relationships
            if has_prev and has_next:
                return -1   # 中间位置
            elif not has_prev and has_next:
                return 1    # 第一个 chunk
            elif has_prev and not has_next:
                return -2   # 最后一个 chunk
            elif not has_prev and not has_next:
                return 0    # 唯一的 chunk（文档很短）
    except Exception:
        pass
    return -1


class LlamaIndexRetriever(BaseRetriever):
    """
    LlamaIndex 检索器。

    使用 LlamaIndex retriever 从 FAISS 索引中获取 Node 列表，
    不调用 LLM，不生成回答。仅做向量检索。
    """

    def __init__(self) -> None:
        """初始化检索器。"""
        self._builder = LlamaIndexBuilder()

    # ------------------------------------------------------------------
    # 核心检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行检索并返回结构化结果。

        流程：
          1. 确保索引已加载（如未加载则从磁盘加载）
          2. 使用 index.as_retriever().retrieve() 获取 Node 列表
          3. 将 Node 转换为与 LangChain 一致的标准 Dict 格式

        Args:
            query: 查询问题
            top_k: 返回结果数量
            score_threshold: 相似度阈值（低于此值过滤）

        Returns:
            检索结果字典
        """
        logger.info(f"[LlamaIndex] 检索请求: query='{query[:80]}...', top_k={top_k}")

        # 确保索引已加载
        if not self._builder.is_loaded:
            if not self._builder.load_index():
                logger.error("LlamaIndex 索引加载失败，返回空结果")
                return {
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "message": "LlamaIndex 索引不存在，请先上传文档并构建索引。",
                }

        # 使用 retriever 检索（不调用 LLM）
        with Timer("LlamaIndex 检索") as timer:
            retriever = self._builder.index.as_retriever(
                similarity_top_k=top_k,
            )
            nodes: list = retriever.retrieve(query)

        # 将 Node 转换为标准 Dict 格式
        results: List[Dict[str, Any]] = []
        for node in nodes:
            # LlamaIndex Node 的相似度分数（clamp 到 [0, 1]）
            raw_score: float = node.score or 0.0
            similarity: float = round(max(0.0, min(1.0, raw_score)), 4)

            # 阈值过滤
            if similarity < score_threshold:
                continue

            # 提取元数据
            metadata: Dict[str, Any] = dict(node.metadata) if node.metadata else {}

            # 提取 Node Relationships（LlamaIndex 特有）
            relationships: Dict[str, Any] = {}
            inner = node.node
            if hasattr(inner, 'relationships') and inner.relationships:
                from llama_index.core.schema import NodeRelationship
                for rel_type, rel_info in inner.relationships.items():
                    if isinstance(rel_type, NodeRelationship):
                        rel_name = rel_type.value  # "1"=SOURCE, "2"=PREVIOUS, "3"=NEXT
                    else:
                        rel_name = str(rel_type)
                    relationships[rel_name] = {
                        "node_id": rel_info.node_id[:16] + "..." if rel_info.node_id else None,
                        "node_type": str(rel_info.node_type) if rel_info.node_type else None,
                    }

            # 构建完整元数据
            metadata.update({
                "node_id": node.node_id,
                "source": metadata.get("file_name", "未知来源"),
                "chunk_index": _extract_chunk_order(inner),  # 从 relationships 推断顺序
                "relationships": relationships,
            })

            results.append({
                "page_content": node.text or node.get_content(),
                "metadata": metadata,
                "similarity_score": similarity,
            })

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "retrieval_time": round(timer.elapsed, 4),
            "top_k": top_k,
        }

    # ------------------------------------------------------------------
    # 格式化
    # ------------------------------------------------------------------

    def format_retrieval_for_llm(self, retrieval_result: Dict[str, Any]) -> str:
        """
        将检索结果格式化为 LLM 可读文本。
        包含 Node 关系信息（SOURCE / PREVIOUS / NEXT）。

        Args:
            retrieval_result: retrieve() 返回的结果字典

        Returns:
            格式化文本
        """
        results: List[Dict[str, Any]] = retrieval_result.get("results", [])
        if not results:
            return "根据当前知识库无法确定。"

        formatted_parts: List[str] = []
        for i, result in enumerate(results, 1):
            metadata: Dict[str, Any] = result.get("metadata", {})
            source: str = metadata.get("source", "未知来源")
            node_id: str = metadata.get("node_id", "")
            similarity: float = result.get("similarity_score", 0.0)
            content: str = result.get("page_content", "")
            chunk_index: int = metadata.get("chunk_index", -1)
            relationships: Dict[str, Any] = metadata.get("relationships", {})

            # 构建关系描述
            rel_lines: str = ""
            if relationships:
                # SOURCE 关系 → 来自哪个文档
                if "1" in relationships:  # NodeRelationship.SOURCE = "1"
                    rel_lines += f"  └─ SOURCE → Document: {relationships['1'].get('node_id', '?')}\n"
                # PREVIOUS 关系 → 前一个 chunk
                if "2" in relationships:  # NodeRelationship.PREVIOUS = "2"
                    rel_lines += f"  └─ PREVIOUS → Node: {relationships['2'].get('node_id', '?')}\n"
                # NEXT 关系 → 后一个 chunk
                if "3" in relationships:  # NodeRelationship.NEXT = "3"
                    rel_lines += f"  └─ NEXT → Node: {relationships['3'].get('node_id', '?')}\n"

            # chunk 位置描述
            position_desc: str = {
                1: "（文档开头片段）",
                -2: "（文档末尾片段）",
                0: "（唯一片段）",
            }.get(chunk_index, "（中间片段）")

            formatted_parts.append(
                f"【来源 {i}】{position_desc}\n"
                f"  引擎: LlamaIndex\n"
                f"  文档: {source}\n"
                f"  Node ID: {node_id[:20]}...\n"
                f"  相似度: {similarity:.2%}\n"
                + (rel_lines if rel_lines else "") +
                f"  内容: {content}\n"
            )

        return "\n".join(formatted_parts)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息。"""
        return self._builder.get_stats()

    def rebuild_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        用新数据重建索引。

        注意：LlamaIndex 的 rebuild 流程由 KnowledgeService 编排
        （SimpleDirectoryReader → build_index），此方法预留。
        """
        logger.info("LlamaIndexRetriever: rebuild_index() 由 KnowledgeService 编排")

    def delete_index(self) -> None:
        """删除当前索引。"""
        self._builder.delete_index()
