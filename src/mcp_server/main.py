"""
Hierarchy-RAG MCP Server

暴露三个 MCP tools：
    ingest_document    — 将 MD/PDF 文档 ingest 到 Qdrant
    query_collection   — 对指定 collection 做语义检索
    manage_collection  — 创建/删除/列出 collection

启动方式：
    python src/mcp_server/main.py --host 0.0.0.0 --port 9620
    python src/mcp_server/main.py --stdio   # stdio 模式（Claude Desktop / OpenClaw）

环境变量：
    QDRANT_HOST / QDRANT_PORT / QDRANT_URL / QDRANT_API_KEY
    EMBEDDER_PROVIDER / OLLAMA_HOST / OLLAMA_EMBED_MODEL
    OPENAI_API_KEY / OPENAI_BASE_URL
    PDF_PARSER          默认 mineru
    MINERU_CONDA_ENV    默认 mineru
    SCHEMAS_DIR         schemas 目录路径，默认 ./schemas
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    from mcp import types as mcp_types
except ImportError as e:
    print(
        f"[ERROR] mcp package not found: {e}\n"
        "Install with: pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

from src.schema import SchemaLoader
from src.store.qdrant import QdrantStore
from src.ingestor.pipeline import IngestPipeline
from src.embedder import load_embedder

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hierarchy-rag")

# ------------------------------------------------------------------
# MCP Server 实例
# ------------------------------------------------------------------

app = Server("hierarchy-rag")


def _get_store() -> QdrantStore:
    return QdrantStore()


def _get_schemas_dir() -> Path:
    return Path(os.environ.get("SCHEMAS_DIR", Path(__file__).parent.parent.parent / "schemas"))


# ------------------------------------------------------------------
# Tool: ingest_document
# ------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="ingest_document",
            description=(
                "Ingest a Markdown or PDF document into a Qdrant collection for RAG. "
                "The document is chunked, embedded, and stored with metadata. "
                "Requires a schema.yaml file in the schemas/ directory for the target collection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the .md, .markdown, .txt, or .pdf file.",
                    },
                    "collection": {
                        "type": "string",
                        "description": (
                            "Target Qdrant collection name. "
                            "A corresponding schemas/<collection>.yaml must exist."
                        ),
                    },
                    "schema_file": {
                        "type": "string",
                        "description": (
                            "Optional: absolute path to a custom schema YAML file. "
                            "If omitted, uses schemas/<collection>.yaml."
                        ),
                    },
                    "extra_metadata": {
                        "type": "object",
                        "description": "Optional key-value pairs to attach to every chunk.",
                    },
                    "pdf_parser": {
                        "type": "string",
                        "description": "PDF parser to use: 'mineru' (default) or 'pymupdf'.",
                        "enum": ["mineru", "pymupdf"],
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true (default), delete existing chunks from this file before ingesting.",
                    },
                },
                "required": ["file_path", "collection"],
            },
        ),
        mcp_types.Tool(
            name="query_collection",
            description=(
                "Perform semantic search on a Qdrant collection. "
                "The embedding model is automatically read from the collection's schema.yaml "
                "to guarantee it matches the model used during ingest. "
                "Returns the most relevant text chunks with their metadata and similarity scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in natural language.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Target Qdrant collection name to search in.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                        "default": 5,
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1). Results below this are filtered out.",
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional metadata filters as key-value pairs. "
                            "E.g. {\"heading_level\": 2} to only return level-2 section chunks."
                        ),
                    },
                },
                "required": ["query", "collection"],
            },
        ),
        mcp_types.Tool(
            name="manage_collection",
            description=(
                "Manage Qdrant collections: list all, get info, or delete a specific collection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform.",
                        "enum": ["list", "info", "delete"],
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection name (required for 'info' and 'delete').",
                    },
                },
                "required": ["action"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
    try:
        if name == "ingest_document":
            result = await _handle_ingest(arguments)
        elif name == "query_collection":
            result = await _handle_query(arguments)
        elif name == "manage_collection":
            result = await _handle_manage(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.exception("Tool '%s' raised an exception: %s", name, e)
        result = {"error": str(e)}

    return [mcp_types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ------------------------------------------------------------------
# Tool handlers
# ------------------------------------------------------------------

async def _handle_ingest(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["file_path"]
    collection = args["collection"]
    schema_file = args.get("schema_file")
    extra_metadata = args.get("extra_metadata")
    pdf_parser = args.get("pdf_parser", os.environ.get("PDF_PARSER", "mineru"))
    overwrite = args.get("overwrite", True)

    # 加载 schema
    if schema_file:
        schema = SchemaLoader.load_file(schema_file)
    else:
        schema = SchemaLoader.load(collection, schemas_dir=_get_schemas_dir())

    pipeline = IngestPipeline(
        schema=schema,
        store=_get_store(),
        pdf_parser=pdf_parser,
        overwrite=overwrite,
    )

    result = pipeline.ingest(file_path, extra_metadata=extra_metadata)
    return result.to_dict()


async def _handle_query(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    collection = args["collection"]
    top_k = args.get("top_k", 5)
    score_threshold = args.get("score_threshold")
    filters = args.get("filters")

    # Embedder 固定从 schema.yaml 读取，禁止外部覆盖。
    # 原因：ingest 和 query 必须使用完全相同的 embedding 模型，
    # 否则向量空间不同，查询结果毫无意义。
    schema = SchemaLoader.load(collection, schemas_dir=_get_schemas_dir())
    embedder = load_embedder(
        provider=schema.embedder.provider,
        config={"model": schema.embedder.model, **schema.embedder.extra},
    )
    query_vector = embedder.embed_one(query)

    store = _get_store()
    results = store.search(
        collection=collection,
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
        filters=filters,
    )

    return {
        "query": query,
        "collection": collection,
        "embedder": f"{schema.embedder.provider}/{schema.embedder.model}",
        "results_count": len(results),
        "results": results,
    }


async def _handle_manage(args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    collection = args.get("collection")
    store = _get_store()

    if action == "list":
        collections = store.list_collections()
        return {"collections": collections, "count": len(collections)}

    if action == "info":
        if not collection:
            return {"error": "collection is required for action='info'"}
        try:
            return store.collection_info(collection)
        except Exception as e:
            return {"error": str(e)}

    if action == "delete":
        if not collection:
            return {"error": "collection is required for action='delete'"}
        deleted = store.delete_collection(collection)
        return {
            "collection": collection,
            "deleted": deleted,
            "message": f"Collection '{collection}' deleted." if deleted else "Collection not found.",
        }

    return {"error": f"Unknown action: {action}"}


# ------------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchy-RAG MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host (SSE mode)")
    parser.add_argument("--port", type=int, default=9620, help="HTTP server port (SSE mode)")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport (for Claude Desktop / OpenClaw)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if args.stdio:
        logger.info("Starting Hierarchy-RAG MCP Server in stdio mode")
        import asyncio
        asyncio.run(_run_stdio())
    else:
        logger.info("Starting Hierarchy-RAG MCP Server on %s:%d", args.host, args.port)
        import asyncio
        asyncio.run(_run_sse(args.host, args.port))


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def _run_sse(host: str, port: int) -> None:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount

    sse = SseServerTransport("/messages")

    async def handle_sse(request: Any) -> Any:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse.handle_post_message),
        ]
    )

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    main()
