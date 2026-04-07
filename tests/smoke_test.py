"""
Smoke test — 验证完整的 ingest → query 流程。

运行方式：
    conda run -n mcp-server python tests/smoke_test.py

要求：
    - Qdrant 在 localhost:6333 运行
    - Ollama 在 localhost:11434 运行，已拉取 bge-m3
    - 在项目根目录运行
"""

import sys
import os

# 保证从项目根目录可以 import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import SchemaLoader
from src.ingestor import IngestPipeline
from src.store.qdrant import QdrantStore

COLLECTION = "smoke_test"
TEST_MD = os.path.join(os.path.dirname(__file__), "../examples/sample_md/test_doc.md")


def _green(s): return f"\033[92m{s}\033[0m"
def _red(s):   return f"\033[91m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"


def check(label, cond, detail=""):
    status = _green("PASS") if cond else _red("FAIL")
    print(f"  [{status}] {label}" + (f"  →  {detail}" if detail else ""))
    return cond


def main():
    all_passed = True
    print(_bold("\n=== Hierarchy-RAG Smoke Test ===\n"))

    # ── Step 1: Qdrant 连通性 ─────────────────────────────────
    print(_bold("1. Qdrant connectivity"))
    try:
        store = QdrantStore()
        cols = store.list_collections()
        all_passed &= check("Qdrant reachable", True, f"collections={cols}")
    except Exception as e:
        check("Qdrant reachable", False, str(e))
        print(_red("  Cannot continue without Qdrant. Exiting."))
        sys.exit(1)

    # ── Step 2: Schema 加载 ───────────────────────────────────
    print(_bold("\n2. Schema loading"))
    try:
        # 使用临时 inline schema（不依赖 schemas/ 目录文件）
        from src.schema.loader import SchemaLoader
        schema = SchemaLoader.load_dict({
            "collection": COLLECTION,
            "chunker": "heading",
            "chunker_config": {"max_chunk_size": 800, "min_chunk_size": 30},
            "embedder": {"provider": "ollama", "model": "bge-m3"},
            "distance": "Cosine",
            "metadata_fields": [
                {"name": "title",   "type": "string", "source": "frontmatter", "default": ""},
                {"name": "tags",    "type": "list",   "source": "frontmatter", "default": []},
                {"name": "created", "type": "string", "source": "frontmatter", "default": None},
            ],
        })
        all_passed &= check("Schema parsed", True, f"collection={schema.collection}")
    except Exception as e:
        check("Schema parsed", False, str(e))
        sys.exit(1)

    # ── Step 3: Chunker ───────────────────────────────────────
    print(_bold("\n3. Chunker (HeadingChunker)"))
    try:
        from src.chunker import load_chunker
        chunker = load_chunker("heading", config={"max_chunk_size": 800, "min_chunk_size": 30})
        with open(TEST_MD, encoding="utf-8") as f:
            raw_text = f.read()
        from src.schema.extractor import MetadataExtractor
        extractor = MetadataExtractor(schema)
        frontmatter, clean_text = extractor.extract_frontmatter(raw_text)
        chunks = chunker.chunk(clean_text, base_metadata={"source_file": TEST_MD})
        all_passed &= check("Chunks generated", len(chunks) > 0, f"{len(chunks)} chunks")
        print(f"    Sample chunk[0]: {repr(chunks[0].text[:80])}")
        print(f"    metadata: {chunks[0].metadata}")
    except Exception as e:
        check("Chunker", False, str(e))
        sys.exit(1)

    # ── Step 4: Embedder ──────────────────────────────────────
    print(_bold("\n4. Embedder (Ollama bge-m3)"))
    try:
        from src.embedder import load_embedder
        embedder = load_embedder("ollama", config={"model": "bge-m3"})
        dim = embedder.dimension
        all_passed &= check("Dimension detected", dim > 0, f"dim={dim}")
        test_vec = embedder.embed_one("test sentence for embedding")
        all_passed &= check("embed_one works", len(test_vec) == dim, f"len={len(test_vec)}")
    except Exception as e:
        check("Embedder", False, str(e))
        sys.exit(1)

    # ── Step 5: 完整 ingest pipeline ─────────────────────────
    print(_bold("\n5. IngestPipeline (MD file → Qdrant)"))
    try:
        # 先清理可能的旧 collection
        if COLLECTION in store.list_collections():
            store.delete_collection(COLLECTION)
            print(f"    (cleaned up old collection '{COLLECTION}')")

        pipeline = IngestPipeline(schema, store=store)
        result = pipeline.ingest(TEST_MD)
        all_passed &= check("Ingest success",   result.success, result.error or "")
        all_passed &= check("Chunks stored > 0", result.chunks_stored > 0,
                            f"{result.chunks_stored}/{result.chunks_total} chunks")
        print(f"    elapsed: {result.elapsed_seconds:.1f}s  "
              f"chunker: {result.chunker_used}  embedder: {result.embedder_used}")
    except Exception as e:
        check("IngestPipeline", False, str(e))
        sys.exit(1)

    # ── Step 6: Query ─────────────────────────────────────────
    print(_bold("\n6. Query (semantic search)"))
    try:
        query = "Qdrant 向量数据库的 collection 管理"
        q_vec = embedder.embed_one(query)
        results = store.search(COLLECTION, q_vec, top_k=3)
        all_passed &= check("Results returned", len(results) > 0, f"{len(results)} results")
        if results:
            top = results[0]
            all_passed &= check("Score in range", 0 <= top["score"] <= 1, f"score={top['score']:.4f}")
            print(f"\n    Query: {query!r}")
            for i, r in enumerate(results, 1):
                print(f"    [{i}] score={r['score']:.4f}  heading={r['metadata'].get('heading', '?')!r}")
                print(f"        {r['text'][:120].replace(chr(10), ' ')!r}")
    except Exception as e:
        check("Query", False, str(e))

    # ── Step 7: ingest_chunks 直接接口 ────────────────────────
    print(_bold("\n7. ingest_chunks (bypass parser/chunker)"))
    try:
        from src.chunker.base import Chunk
        custom_chunks = [
            Chunk(text="自定义 chunk 一：用于测试 ingest_chunks 接口。",
                  metadata={"source_file": "custom_test", "page": 1}),
            Chunk(text="自定义 chunk 二：验证直接传入 chunk 列表跳过 parser 和 chunker。",
                  metadata={"source_file": "custom_test", "page": 2}),
        ]
        result2 = pipeline.ingest_chunks(custom_chunks, source_label="custom_test")
        all_passed &= check("ingest_chunks success", result2.success, result2.error or "")
        all_passed &= check("Custom chunks stored",  result2.chunks_stored == 2,
                            f"{result2.chunks_stored} stored")
    except Exception as e:
        check("ingest_chunks", False, str(e))

    # ── Step 8: collection info ───────────────────────────────
    print(_bold("\n8. Collection info"))
    try:
        info = store.collection_info(COLLECTION)
        all_passed &= check("Collection exists", info["points_count"] > 0,
                            f"points={info['points_count']}")
    except Exception as e:
        check("Collection info", False, str(e))

    # ── Cleanup ───────────────────────────────────────────────
    print(_bold("\n9. Cleanup"))
    try:
        store.delete_collection(COLLECTION)
        check("Test collection deleted", COLLECTION not in store.list_collections())
    except Exception as e:
        check("Cleanup", False, str(e))

    # ── Summary ───────────────────────────────────────────────
    print()
    if all_passed:
        print(_green(_bold("All checks passed. Hierarchy-RAG is working correctly.")))
    else:
        print(_red(_bold("Some checks failed. See output above.")))
        sys.exit(1)


if __name__ == "__main__":
    main()
