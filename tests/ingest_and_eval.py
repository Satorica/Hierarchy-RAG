"""
Ingest + Retrieval Evaluation
------------------------------
1. 用 MineruCloudParser 解析 sample_pdf/Attention Is All You Need.pdf
2. 按章节（HeadingChunker）分块，ingest 进 dl_papers collection
3. 用 10 个典型问题测试召回效果，打印 top-k 结果

运行：
    export MINERU_API_TOKEN=your_token_here
    conda run -n mcp-server python tests/ingest_and_eval.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import SchemaLoader
from src.ingestor import IngestPipeline
from src.store.qdrant import QdrantStore
from src.embedder import load_embedder

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../examples/sample_pdf/Attention Is All You Need.pdf"
)
COLLECTION = "dl_papers"
TOP_K = 5

# 测试用的问题集：覆盖论文各核心章节
EVAL_QUERIES = [
    # 架构 & 模型结构
    ("Q1", "What is the overall architecture of the Transformer model?"),
    ("Q2", "How does multi-head attention work?"),
    ("Q3", "What is the purpose of positional encoding in the Transformer?"),
    # 注意力机制
    ("Q4", "How is scaled dot-product attention computed?"),
    ("Q5", "Why does the paper use scaled dot-product instead of additive attention?"),
    # 训练 & 实验
    ("Q6", "What optimizer and learning rate schedule was used for training?"),
    ("Q7", "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?"),
    # 消融实验
    ("Q8", "What did the ablation study show about the number of attention heads?"),
    # 计算复杂度
    ("Q9", "How does the computational complexity of self-attention compare to recurrent layers?"),
    # 泛化能力
    ("Q10", "How did the Transformer perform on English constituency parsing?"),
]


def _sep(char="─", n=80):
    print(char * n)


def main():
    # ── Step 1: 检查 Token ────────────────────────────────────
    if not os.environ.get("MINERU_API_TOKEN"):
        print("ERROR: MINERU_API_TOKEN not set.")
        print("Run: export MINERU_API_TOKEN=your_token_here")
        sys.exit(1)

    # ── Step 2: 加载 schema ───────────────────────────────────
    schema = SchemaLoader.load("dl_papers")
    store = QdrantStore()
    embedder = load_embedder("ollama", config={"model": "bge-m3"})

    # ── Step 3: Ingest（如果 collection 已有数据则跳过重新解析）
    _sep()
    if COLLECTION in store.list_collections():
        info = store.collection_info(COLLECTION)
        pts = info["points_count"]
        print(f"Collection '{COLLECTION}' already exists with {pts} points.")
        ans = input("Re-ingest? [y/N] ").strip().lower()
        if ans != "y":
            print("Skipping ingest, using existing data.")
        else:
            _do_ingest(schema, store)
    else:
        _do_ingest(schema, store)

    # ── Step 4: 召回评估 ──────────────────────────────────────
    _sep()
    print(f"{'RETRIEVAL EVALUATION':^80}")
    print(f"Collection: {COLLECTION}  |  Embedder: bge-m3  |  top_k={TOP_K}")
    _sep()

    score_sum = 0.0
    for qid, query in EVAL_QUERIES:
        q_vec = embedder.embed_one(query)
        results = store.search(COLLECTION, q_vec, top_k=TOP_K)

        print(f"\n{qid}: {query}")
        print(f"{'─'*76}")
        for rank, hit in enumerate(results, 1):
            score = hit["score"]
            heading = hit["metadata"].get("heading", "?")
            heading_path = hit["metadata"].get("heading_path", "")
            has_formula = hit["metadata"].get("has_formula", False)
            has_table = hit["metadata"].get("has_table", False)
            flags = ("📐" if has_formula else "") + ("📊" if has_table else "")
            preview = hit["text"][:150].replace("\n", " ")
            print(f"  [{rank}] score={score:.4f}  {flags}  /{heading_path}/")
            print(f"       {preview!r}")

        if results:
            score_sum += results[0]["score"]

    _sep()
    avg_top1 = score_sum / len(EVAL_QUERIES)
    print(f"Average top-1 similarity score: {avg_top1:.4f}")
    print(f"(Higher = embedding space aligns better with query intent)")
    _sep()


def _do_ingest(schema, store):
    print(f"Ingesting: {os.path.basename(PDF_PATH)}")
    print("Using MineruCloudParser (vlm model)...")
    _sep("─")

    # extra_metadata：覆盖 schema.yaml 中的 static 占位值
    extra = {
        "topic": "transformer,attention,NLP,deep_learning",
        "owner": "chen",
        "published_year": 2017,
        "venue": "NeurIPS",
        "arxiv_id": "1706.03762",
        "paper_title": "Attention Is All You Need",
    }

    pipeline = IngestPipeline(
        schema=schema,
        store=store,
        pdf_parser="mineru_cloud",
        overwrite=True,
    )

    result = pipeline.ingest(PDF_PATH, extra_metadata=extra)

    if not result.success:
        print(f"ERROR: Ingest failed: {result.error}")
        sys.exit(1)

    print(f"\nIngest complete:")
    print(f"  chunks:   {result.chunks_stored}/{result.chunks_total}")
    print(f"  elapsed:  {result.elapsed_seconds:.1f}s")
    print(f"  parser:   {result.parser_used}")
    print(f"  chunker:  {result.chunker_used}")
    print(f"  embedder: {result.embedder_used}")


if __name__ == "__main__":
    main()
