# Hierarchy-RAG

Private RAG system with hierarchical chunking, customizable metadata schemas, and MCP server support.

## Architecture

```
Input (MD / PDF)
    │
    ▼
Parser (MinerU / PyMuPDF)         ← BasePDFParser protocol
    │  PDF → Markdown
    ▼
Chunker (Heading / Fixed / Para)  ← BaseChunker protocol (user-extensible)
    │  Markdown → Chunks + metadata
    ▼
MetadataExtractor                 ← driven by schema.yaml
    │  enrich each chunk with custom fields
    ▼
Embedder (Ollama / OpenAI)        ← BaseEmbedder protocol
    │  text → vectors
    ▼
QdrantStore                       ← collection per topic
    │
    ▼
MCP Server (3 tools)
    ingest_document / query_collection / manage_collection
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# For PDF parsing via MinerU (separate conda env):
# conda create -n mineru python=3.11
# conda activate mineru && pip install mineru==3.0.8

# Start Qdrant (Docker):
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Pull embedding model

```bash
ollama pull bge-m3
```

### 3. Define a collection schema

```bash
cp schemas/research_articles.yaml schemas/my_collection.yaml
# Edit my_collection.yaml to set collection name, chunker, metadata fields
```

### 4. Start the MCP Server

```bash
# SSE mode (for HTTP clients)
python src/mcp_server/main.py --host 0.0.0.0 --port 9620

# stdio mode (for Claude Desktop / OpenClaw)
python src/mcp_server/main.py --stdio
```

### 5. Ingest documents

Via MCP tool call:
```json
{
  "tool": "ingest_document",
  "arguments": {
    "file_path": "/path/to/paper.pdf",
    "collection": "research_articles"
  }
}
```

Or directly in Python:
```python
from src.schema import SchemaLoader
from src.ingestor import IngestPipeline

schema = SchemaLoader.load("research_articles")
pipeline = IngestPipeline(schema)
result = pipeline.ingest("/path/to/paper.md")
print(result.to_dict())
```

### 6. Query

```json
{
  "tool": "query_collection",
  "arguments": {
    "query": "What is the main contribution of Chapter 3?",
    "collection": "research_articles",
    "top_k": 5
  }
}
```

---

## Custom Chunker

Inherit `BaseChunker` and register in `schema.yaml`:

```python
# my_project/my_chunker.py
from src.chunker.base import BaseChunker, Chunk

class SentenceChunker(BaseChunker):
    def chunk(self, text: str, base_metadata: dict) -> list[Chunk]:
        sentences = text.split(". ")
        return [Chunk(text=s, metadata=base_metadata) for s in sentences if s.strip()]
```

```yaml
# schemas/my_collection.yaml
chunker: my_project.my_chunker.SentenceChunker
```

## Custom Embedder

```python
from src.embedder.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
    @property
    def dimension(self) -> int:
        return 768
```

```bash
EMBEDDER_PROVIDER=my_project.my_embedder.MyEmbedder python src/mcp_server/main.py --stdio
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_URL` | — | Full URL (overrides HOST:PORT), use for Qdrant Cloud |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `EMBEDDER_PROVIDER` | `ollama` | `ollama` / `openai` / custom module path |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Ollama embedding model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | — | OpenAI-compatible API base URL |
| `PDF_PARSER` | `mineru` | `mineru` / `pymupdf` |
| `MINERU_CONDA_ENV` | `mineru` | conda env for MinerU |
| `SCHEMAS_DIR` | `./schemas` | Path to schemas directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
src/
├── chunker/        BaseChunker + HeadingChunker, FixedSizeChunker, ParagraphChunker
├── embedder/       BaseEmbedder + OllamaEmbedder, OpenAIEmbedder
├── parser/         BasePDFParser + MineruParser, PyMuPDFParser
├── schema/         CollectionSchema, SchemaLoader, MetadataExtractor
├── store/          QdrantStore
├── ingestor/       IngestPipeline
└── mcp_server/     main.py (3 MCP tools)
schemas/
├── research_articles.yaml
└── personal_notes.yaml
```
