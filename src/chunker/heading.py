"""
HeadingChunker — 按 Markdown 标题层级分块。

每个 chunk 对应一个标题节（从该标题到下一个同级/父级标题之间的内容）。
每个 chunk 自动在 metadata 中记录完整的父级标题路径（heading_path）。

schema.yaml 配置示例：
    chunker: heading
    chunker_config:
      max_chunk_size: 1500      # 单个 chunk 的最大字符数（超出时按段落二次切割）
      min_chunk_size: 50        # 忽略过短的 chunk（通常是空节）
      inherit_parent_headings: true  # 是否在 metadata 中保留完整标题路径
      include_heading_in_text: true  # chunk text 中是否包含标题行本身
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseChunker, Chunk


class HeadingChunker(BaseChunker):
    """按 Markdown 标题（#/##/###...）层级切割文档。"""

    DEFAULT_MAX_CHUNK_SIZE = 1500
    DEFAULT_MIN_CHUNK_SIZE = 50

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.max_chunk_size: int = self.config.get("max_chunk_size", self.DEFAULT_MAX_CHUNK_SIZE)
        self.min_chunk_size: int = self.config.get("min_chunk_size", self.DEFAULT_MIN_CHUNK_SIZE)
        self.inherit_parent: bool = self.config.get("inherit_parent_headings", True)
        self.include_heading: bool = self.config.get("include_heading_in_text", True)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        sections = self._split_by_headings(text)
        chunks: list[Chunk] = []

        for section in sections:
            heading_path = section["heading_path"]
            content = section["content"]

            if len(content.strip()) < self.min_chunk_size:
                continue

            # 超长节做二次切割
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_oversized(content, heading_path, base_metadata)
                chunks.extend(sub_chunks)
            else:
                meta = {**base_metadata, **self._build_heading_meta(heading_path)}
                chunks.append(Chunk(text=content.strip(), metadata=meta))

        return chunks

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _split_by_headings(self, text: str) -> list[dict]:
        """将 Markdown 文本按标题切分为节列表。"""
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))

        if not matches:
            # 没有任何标题，整篇文档作为一个 chunk
            return [{"heading_path": [], "content": text}]

        sections = []
        # 维护当前标题栈 [(level, title), ...]
        heading_stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # 更新标题栈
            # 弹出所有 level >= 当前 level 的条目（保留更高层级的祖先）
            heading_stack = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))

            content_start = match.end()
            content_body = text[content_start:end]

            if self.include_heading:
                content = match.group(0) + "\n" + content_body
            else:
                content = content_body

            sections.append({
                "heading_path": [t for _, t in heading_stack],
                "content": content,
            })

        # 文档开头（第一个标题之前）的内容
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, {"heading_path": [], "content": preamble})

        return sections

    def _build_heading_meta(self, heading_path: list[str]) -> dict:
        meta: dict[str, Any] = {}
        if self.inherit_parent:
            meta["heading_path"] = " > ".join(heading_path) if heading_path else ""
        if heading_path:
            meta["heading"] = heading_path[-1]
            meta["heading_level"] = len(heading_path)
        return meta

    def _split_oversized(
        self,
        content: str,
        heading_path: list[str],
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """将超长节按段落进行二次切割。"""
        paragraphs = re.split(r"\n{2,}", content)
        chunks: list[Chunk] = []
        buffer = ""
        part_idx = 0

        for para in paragraphs:
            if len(buffer) + len(para) + 2 > self.max_chunk_size and buffer:
                meta = {
                    **base_metadata,
                    **self._build_heading_meta(heading_path),
                    "split_part": part_idx,
                }
                chunks.append(Chunk(text=buffer.strip(), metadata=meta))
                buffer = para
                part_idx += 1
            else:
                buffer = buffer + "\n\n" + para if buffer else para

        if buffer.strip():
            meta = {
                **base_metadata,
                **self._build_heading_meta(heading_path),
                "split_part": part_idx,
            }
            chunks.append(Chunk(text=buffer.strip(), metadata=meta))

        return chunks
