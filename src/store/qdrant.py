"""
QdrantStore — Qdrant 向量数据库封装。

设计原则：collection per topic（每个知识领域独立 collection）。
支持本地 Qdrant 实例和 Qdrant Cloud。

环境变量：
    QDRANT_HOST      默认 localhost
    QDRANT_PORT      默认 6333
    QDRANT_API_KEY   可选，Qdrant Cloud 时使用
    QDRANT_URL       可选，完整 URL（如 https://xxx.qdrant.io），优先于 HOST:PORT

依赖：
    pip install qdrant-client
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from src.chunker.base import Chunk

_QDRANT_AVAILABLE = False
try:
    from qdrant_client import QdrantClient as _QdrantClient  # type: ignore[import]
    _QDRANT_AVAILABLE = True
except ImportError:
    pass


class QdrantStore:
    """
    封装 Qdrant 操作：创建集合、写入向量、语义检索。

    Args:
        host: Qdrant host（优先读取环境变量 QDRANT_HOST）。
        port: Qdrant port（优先读取环境变量 QDRANT_PORT）。
        api_key: Qdrant Cloud API key（可选）。
        url: 完整 URL，优先于 host:port（适用于 Qdrant Cloud）。
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        url: str | None = None,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        _url = url or os.environ.get("QDRANT_URL")
        _host = host or os.environ.get("QDRANT_HOST", "localhost")
        _port = port or int(os.environ.get("QDRANT_PORT", "6333"))
        _api_key = api_key or os.environ.get("QDRANT_API_KEY")

        if _url:
            self._client = _QdrantClient(url=_url, api_key=_api_key)  # type: ignore[name-defined]
        else:
            self._client = _QdrantClient(host=_host, port=_port, api_key=_api_key)  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        collection: str,
        dimension: int,
        distance: str = "Cosine",
    ) -> bool:
        """
        确保 collection 存在；若不存在则创建。

        Args:
            collection: collection 名称。
            dimension: 向量维度。
            distance: 距离度量（Cosine / Dot / Euclid）。

        Returns:
            True 表示新建，False 表示已存在。
        """
        from qdrant_client.http import models as m

        existing = [c.name for c in self._client.get_collections().collections]
        if collection in existing:
            return False

        distance_map = {
            "Cosine": m.Distance.COSINE,
            "Dot": m.Distance.DOT,
            "Euclid": m.Distance.EUCLID,
        }
        dist = distance_map.get(distance, m.Distance.COSINE)

        self._client.create_collection(
            collection_name=collection,
            vectors_config=m.VectorParams(size=dimension, distance=dist),
        )
        return True

    def list_collections(self) -> list[str]:
        """返回所有 collection 名称列表。"""
        return [c.name for c in self._client.get_collections().collections]

    def delete_collection(self, collection: str) -> bool:
        """
        删除指定 collection。

        Returns:
            True 表示删除成功，False 表示 collection 不存在。
        """
        if collection not in self.list_collections():
            return False
        self._client.delete_collection(collection_name=collection)
        return True

    def collection_info(self, collection: str) -> dict[str, Any]:
        """返回 collection 的基本信息（点数、向量维度等）。"""
        info = self._client.get_collection(collection_name=collection)
        return {
            "name": collection,
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", None),
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """
        批量写入 chunks 及其向量。

        Args:
            collection: 目标 collection 名称。
            chunks: Chunk 列表（含 metadata）。
            vectors: 与 chunks 一一对应的向量列表。

        Returns:
            成功写入的点数量。
        """
        from qdrant_client.http import models as m

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length, "
                f"got {len(chunks)} chunks and {len(vectors)} vectors."
            )

        points = [
            m.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        # 批量分批写入，避免单次请求过大
        batch_size = 100
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=collection, points=batch)
            total += len(batch)

        return total

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索。

        Args:
            collection: 目标 collection 名称。
            query_vector: 查询向量。
            top_k: 返回最相似的 top-k 结果。
            score_threshold: 最低相似度分数（0~1），低于此的结果过滤掉。
            filters: Qdrant payload 过滤条件，格式为 {"field_name": value}。

        Returns:
            结果列表，每项包含 text、score、metadata 等字段。
        """
        from qdrant_client.http import models as m

        qdrant_filter = None
        if filters:
            conditions = [
                m.FieldCondition(
                    key=key,
                    match=m.MatchValue(value=val),
                )
                for key, val in filters.items()
            ]
            qdrant_filter = m.Filter(must=conditions)  # type: ignore[arg-type]

        results = self._client.search(  # type: ignore[attr-defined]
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),  # type: ignore[union-attr]
                "score": hit.score,  # type: ignore[union-attr]
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},  # type: ignore[union-attr]
                "id": str(hit.id),  # type: ignore[union-attr]
            }
            for hit in results
        ]

    def delete_by_source(self, collection: str, source_file: str) -> int:
        """
        删除某个文件来源的所有 chunks（用于重新 ingest 时清理旧数据）。

        Args:
            collection: 目标 collection 名称。
            source_file: 文件路径（与 metadata.source_file 匹配）。

        Returns:
            删除的点数量。
        """
        from qdrant_client.http import models as m

        result = self._client.delete(
            collection_name=collection,
            points_selector=m.FilterSelector(
                filter=m.Filter(
                    must=[
                        m.FieldCondition(
                            key="source_file",
                            match=m.MatchValue(value=source_file),
                        )
                    ]
                )
            ),
        )
        # qdrant-client 返回的 result 没有直接的 deleted_count，用 operation_id 标识成功
        return 0 if result is None else -1  # -1 表示"已执行但不知道具体数量"
