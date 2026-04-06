"""
FixedSizeChunker — 按固定字符数（带 overlap）滑动窗口切割文档。

schema.yaml 配置示例：
    chunker: fixed
    chunker_config:
      chunk_size: 800       # 每个 chunk 的目标字符数
      overlap: 100          # 相邻 chunk 之间的重叠字符数
      split_on_newline: true  # 优先在换行符处对齐切割边界，避免切断句子
"""

from __future__ import annotations

from typing import Any

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """按固定字符数滑动窗口切割，支持 overlap。"""

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP = 100

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.chunk_size: int = self.config.get("chunk_size", self.DEFAULT_CHUNK_SIZE)
        self.overlap: int = self.config.get("overlap", self.DEFAULT_OVERLAP)
        self.split_on_newline: bool = self.config.get("split_on_newline", True)

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # 尝试在换行符处对齐，避免切断句子
            if self.split_on_newline and end < text_len:
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos + 1  # 包含换行符本身

            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": end,
                }
                chunks.append(Chunk(text=chunk_text, metadata=meta))
                chunk_idx += 1

            # 滑动窗口：下一个 chunk 从 (end - overlap) 开始
            next_start = end - self.overlap
            if next_start <= start:
                # 防止死循环（极端情况：overlap 导致无法前进）
                next_start = start + 1
            start = next_start

        return chunks
