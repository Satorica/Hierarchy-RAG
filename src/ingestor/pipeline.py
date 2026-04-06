"""
IngestPipeline — 串联 parser → chunker → embedder → qdrant store 的完整 ingest 流程。

支持输入类型：
    - .md / .markdown  → 直接读取文本
    - .pdf             → 调用 PDF 解析器转换为 Markdown

用法：
    from src.ingestor.pipeline import IngestPipeline
    from src.schema import SchemaLoader

    schema = SchemaLoader.load("research_articles")
    pipeline = IngestPipeline(schema)
    result = pipeline.ingest("path/to/document.pdf")
    print(result)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.chunker import load_chunker
from src.embedder import load_embedder
from src.parser import load_parser
from src.schema import MetadataExtractor, SchemaLoader
from src.schema.models import CollectionSchema
from src.store.qdrant import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """ingest 操作的结果摘要。"""

    source_file: str
    collection: str
    chunks_total: int
    chunks_stored: int
    elapsed_seconds: float
    chunker_used: str
    embedder_used: str
    parser_used: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_file": self.source_file,
            "collection": self.collection,
            "chunks_total": self.chunks_total,
            "chunks_stored": self.chunks_stored,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "chunker_used": self.chunker_used,
            "embedder_used": self.embedder_used,
            "parser_used": self.parser_used,
            "error": self.error,
        }


class IngestPipeline:
    """
    完整的文档 ingest 管道。

    Args:
        schema: CollectionSchema 实例（含 chunker/embedder/metadata 配置）。
        store: QdrantStore 实例，若为 None 则使用默认连接参数创建。
        pdf_parser: PDF 解析器短名称（"mineru"/"pymupdf"）或完整模块路径。
                    默认 "mineru"，可通过环境变量 PDF_PARSER 覆盖。
        overwrite: 若为 True，ingest 前先删除同一 source_file 的旧数据。
    """

    def __init__(
        self,
        schema: CollectionSchema,
        store: QdrantStore | None = None,
        pdf_parser: str = "mineru",
        overwrite: bool = True,
    ) -> None:
        self.schema = schema
        self.store = store or QdrantStore()
        self.pdf_parser_name = pdf_parser
        self.overwrite = overwrite

        # 初始化各组件
        self.chunker = load_chunker(schema.chunker, config=schema.chunker_config)
        self.embedder = load_embedder(
            provider=schema.embedder.provider,
            config={"model": schema.embedder.model, **schema.embedder.extra},
        )
        self.extractor = MetadataExtractor(schema)

        logger.info(
            "IngestPipeline initialized: collection=%s chunker=%s embedder=%s(%s)",
            schema.collection,
            self.chunker.get_name(),
            self.embedder.get_name(),
            schema.embedder.model,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def ingest(
        self,
        file_path: str,
        extra_metadata: dict[str, Any] | None = None,
        pdf_output_dir: str | None = None,
    ) -> IngestResult:
        """
        将单个文档 ingest 到 Qdrant。

        Args:
            file_path: 文件路径（支持 .md / .markdown / .pdf）。
            extra_metadata: 额外附加到每个 chunk 的 metadata（覆盖自动提取的字段）。
            pdf_output_dir: PDF 解析输出目录（可选）。

        Returns:
            IngestResult 实例。
        """
        t_start = time.time()
        file_path = str(Path(file_path).resolve())
        suffix = Path(file_path).suffix.lower()

        parser_name = ""
        try:
            # 1. 读取 / 解析文档为 Markdown 文本
            if suffix in {".md", ".markdown", ".txt"}:
                markdown_text = Path(file_path).read_text(encoding="utf-8")
            elif suffix == ".pdf":
                markdown_text, parser_name = self._parse_pdf(file_path, pdf_output_dir)
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. Supported: .md, .markdown, .txt, .pdf"
                )

            # 2. 提取 frontmatter 和文档级 metadata
            frontmatter, clean_text = self.extractor.extract_frontmatter(markdown_text)
            doc_meta = self.extractor.extract_doc_metadata(file_path)
            if extra_metadata:
                doc_meta.update(extra_metadata)

            # 3. Chunking
            chunks = self.chunker.chunk(clean_text, base_metadata=doc_meta)
            logger.info("Chunked %s → %d chunks", Path(file_path).name, len(chunks))

            if not chunks:
                return IngestResult(
                    source_file=file_path,
                    collection=self.schema.collection,
                    chunks_total=0,
                    chunks_stored=0,
                    elapsed_seconds=time.time() - t_start,
                    chunker_used=self.chunker.get_name(),
                    embedder_used=self.embedder.get_name(),
                    parser_used=parser_name,
                    error="No chunks generated from document.",
                )

            # 4. 丰富 chunk metadata（按 schema 字段规则）
            for chunk in chunks:
                chunk.metadata = self.extractor.enrich_chunk_metadata(
                    chunk_text=chunk.text,
                    chunk_meta=chunk.metadata,
                    frontmatter=frontmatter,
                    file_path=file_path,
                )

            # 5. Embedding
            texts = [chunk.text for chunk in chunks]
            vectors = self.embedder.embed_texts(texts)
            logger.info("Embedded %d chunks", len(vectors))

            # 6. 确保 collection 存在
            self.store.ensure_collection(
                collection=self.schema.collection,
                dimension=self.embedder.dimension,
                distance=self.schema.distance,
            )

            # 7. 可选：删除旧数据（overwrite 模式）
            if self.overwrite:
                self.store.delete_by_source(self.schema.collection, file_path)

            # 8. 写入 Qdrant
            stored = self.store.upsert_chunks(
                collection=self.schema.collection,
                chunks=chunks,
                vectors=vectors,
            )

            elapsed = time.time() - t_start
            logger.info(
                "Ingest done: %s → %d chunks in %.1fs",
                Path(file_path).name, stored, elapsed,
            )

            return IngestResult(
                source_file=file_path,
                collection=self.schema.collection,
                chunks_total=len(chunks),
                chunks_stored=stored,
                elapsed_seconds=elapsed,
                chunker_used=self.chunker.get_name(),
                embedder_used=self.embedder.get_name(),
                parser_used=parser_name,
            )

        except Exception as e:
            logger.exception("Ingest failed for %s: %s", file_path, e)
            return IngestResult(
                source_file=file_path,
                collection=self.schema.collection,
                chunks_total=0,
                chunks_stored=0,
                elapsed_seconds=time.time() - t_start,
                chunker_used=self.chunker.get_name(),
                embedder_used=self.embedder.get_name(),
                parser_used=parser_name,
                error=str(e),
            )

    def ingest_chunks(
        self,
        chunks: list,
        source_label: str = "<custom>",
    ) -> IngestResult:
        """
        直接接受已经分好的 Chunk 列表，跳过 parser 和 chunker 步骤。

        适用于：
          - 用户自己解析 PDF（不走 MinerU/PyMuPDF）
          - 用户自己实现分块逻辑（不继承 BaseChunker）
          - 处理非文本内容（如多模态场景，chunk.text 可以是图片描述/caption）

        chunks 列表中每个元素只需满足：
          - chunk.text: str         — 用于 embedding 的文本
          - chunk.metadata: dict    — 任意 key-value，直接写入 Qdrant payload

        Args:
            chunks: Chunk 对象列表（src.chunker.base.Chunk），或任何有 .text/.metadata 属性的对象。
            source_label: 写入 metadata["source_file"] 的标识字符串，用于 overwrite 时定位旧数据。

        Returns:
            IngestResult 实例。

        Example::

            from src.chunker.base import Chunk
            from src.ingestor import IngestPipeline
            from src.schema import SchemaLoader

            # 自己解析 PDF，自己分块
            my_chunks = [
                Chunk(text="第一段内容...", metadata={"page": 1, "section": "摘要"}),
                Chunk(text="第二段内容...", metadata={"page": 2, "section": "方法"}),
            ]

            schema = SchemaLoader.load("research_articles")
            pipeline = IngestPipeline(schema)
            result = pipeline.ingest_chunks(my_chunks, source_label="my_paper.pdf")
            print(result.to_dict())
        """
        t_start = time.time()
        try:
            if not chunks:
                return IngestResult(
                    source_file=source_label,
                    collection=self.schema.collection,
                    chunks_total=0,
                    chunks_stored=0,
                    elapsed_seconds=0.0,
                    chunker_used="<external>",
                    embedder_used=self.embedder.get_name(),
                    error="Empty chunks list provided.",
                )

            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_texts(texts)
            logger.info("Embedded %d externally-provided chunks", len(vectors))

            self.store.ensure_collection(
                collection=self.schema.collection,
                dimension=self.embedder.dimension,
                distance=self.schema.distance,
            )

            if self.overwrite:
                self.store.delete_by_source(self.schema.collection, source_label)

            stored = self.store.upsert_chunks(
                collection=self.schema.collection,
                chunks=chunks,
                vectors=vectors,
            )

            return IngestResult(
                source_file=source_label,
                collection=self.schema.collection,
                chunks_total=len(chunks),
                chunks_stored=stored,
                elapsed_seconds=time.time() - t_start,
                chunker_used="<external>",
                embedder_used=self.embedder.get_name(),
            )

        except Exception as e:
            logger.exception("ingest_chunks failed: %s", e)
            return IngestResult(
                source_file=source_label,
                collection=self.schema.collection,
                chunks_total=0,
                chunks_stored=0,
                elapsed_seconds=time.time() - t_start,
                chunker_used="<external>",
                embedder_used=self.embedder.get_name(),
                error=str(e),
            )

    def ingest_many(
        self,
        file_paths: list[str],
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[IngestResult]:
        """
        批量 ingest 多个文档（串行）。

        Returns:
            每个文件对应的 IngestResult 列表。
        """
        results = []
        for fp in file_paths:
            logger.info("Ingesting [%d/%d]: %s", len(results) + 1, len(file_paths), fp)
            result = self.ingest(fp, extra_metadata=extra_metadata)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _parse_pdf(self, pdf_path: str, output_dir: str | None) -> tuple[str, str]:
        """解析 PDF → 返回 (markdown_text, parser_name)。"""
        import os
        parser_provider = os.environ.get("PDF_PARSER", self.pdf_parser_name)
        parser = load_parser(parser_provider)
        result = parser.parse(pdf_path, output_dir=output_dir)
        return result.markdown_text, parser.get_name()
