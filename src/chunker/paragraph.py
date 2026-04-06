"""
ParagraphChunker — 按空行（段落边界）切割文档。

适合叙述性文档（博客、笔记、论文正文等），每个段落作为一个语义单元。
支持合并过短的段落，以及拆分超长段落。

schema.yaml 配置示例：
    chunker: paragraph
    chunker_config:
      min_paragraph_size: 80    # 小于此字符数的段落会与下一段合并
      max_chunk_size: 1200      # 超过此字符数的段落会被进一步切割
      merge_short: true         # 是否合并过短段落
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseChunker, Chunk


class ParagraphChunker(BaseChunker):
    """按段落（空行分隔）切割，支持合并短段落和拆分超长段落。"""

    DEFAULT_MIN_PARA_SIZE = 80
    DEFAULT_MAX_CHUNK_SIZE = 1200

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.min_size: int = self.config.get("min_paragraph_size", self.DEFAULT_MIN_PARA_SIZE)
        self.max_size: int = self.config.get("max_chunk_size", self.DEFAULT_MAX_CHUNK_SIZE)
        self.merge_short: bool = self.config.get("merge_short", True)

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        # 先按两个或以上空行切割为段落
        raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        if self.merge_short:
            raw_paragraphs = self._merge_short_paragraphs(raw_paragraphs)

        chunks: list[Chunk] = []
        chunk_idx = 0

        for para in raw_paragraphs:
            if len(para) > self.max_size:
                # 超长段落按句子切割
                sub_texts = self._split_by_sentence(para)
            else:
                sub_texts = [para]

            for sub in sub_texts:
                if not sub.strip():
                    continue
                meta = {**base_metadata, "chunk_index": chunk_idx}
                chunks.append(Chunk(text=sub.strip(), metadata=meta))
                chunk_idx += 1

        return chunks

    # ------------------------------------------------------------------

    def _merge_short_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """将相邻的短段落合并，直到达到 min_size 或无法再合并。"""
        merged: list[str] = []
        buffer = ""

        for para in paragraphs:
            if buffer:
                candidate = buffer + "\n\n" + para
                if len(buffer) < self.min_size:
                    buffer = candidate
                    continue
            merged.append(buffer) if buffer else None
            buffer = para

        if buffer:
            merged.append(buffer)

        return merged

    def _split_by_sentence(self, text: str) -> list[str]:
        """将超长段落按句号/换行符切割为子块。"""
        # 中英文句子结束符
        sentence_end = re.compile(r"(?<=[。！？.!?])\s+")
        sentences = sentence_end.split(text)

        result: list[str] = []
        buffer = ""

        for sent in sentences:
            if len(buffer) + len(sent) + 1 > self.max_size and buffer:
                result.append(buffer.strip())
                buffer = sent
            else:
                buffer = buffer + " " + sent if buffer else sent

        if buffer.strip():
            result.append(buffer.strip())

        return result if result else [text]
