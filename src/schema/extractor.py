"""
MetadataExtractor — 根据 CollectionSchema 的 metadata_fields 定义，
从文档/chunk 中提取 metadata 值。

支持的 source 类型：
    heading_path  — 直接从 Chunk.metadata["heading_path"] 读取（由 chunker 填写）
    frontmatter   — 从 Markdown YAML frontmatter 中按字段名取值
    regex         — 在 chunk 文本中用正则提取第一个捕获组
    static        — 直接使用 schema 中定义的 value
    filename      — 取文件名（不含扩展名）
    filepath      — 取文件完整路径
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .models import CollectionSchema, MetadataFieldDef


class MetadataExtractor:
    """根据 CollectionSchema 提取 chunk metadata。"""

    def __init__(self, schema: CollectionSchema) -> None:
        self.schema = schema

    def extract_doc_metadata(self, file_path: str) -> dict[str, Any]:
        """
        提取文档级别的 metadata（file-level，在 ingest 时调用一次）。

        Args:
            file_path: 文档文件的绝对路径。

        Returns:
            包含 source_file, collection 等基础字段的字典。
        """
        p = Path(file_path)
        base: dict[str, Any] = {
            "source_file": str(p),
            "filename": p.stem,
            "collection": self.schema.collection,
        }
        return base

    def extract_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        """
        提取并剥离 Markdown frontmatter（--- ... --- 块）。

        Returns:
            (frontmatter_dict, text_without_frontmatter)
        """
        fm_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = fm_pattern.match(text)
        if not match:
            return {}, text

        try:
            fm_data: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            fm_data = {}

        clean_text = text[match.end():]
        return fm_data, clean_text

    def enrich_chunk_metadata(
        self,
        chunk_text: str,
        chunk_meta: dict[str, Any],
        frontmatter: dict[str, Any],
        file_path: str,
    ) -> dict[str, Any]:
        """
        在 chunk 已有的基础 metadata 上，按 schema 中的 metadata_fields 提取并追加字段。

        Args:
            chunk_text: chunk 的文本内容。
            chunk_meta: chunker 生成的原始 metadata（已含 heading_path 等）。
            frontmatter: 文档级 frontmatter dict。
            file_path: 文档文件路径。

        Returns:
            追加了 schema 字段后的新 metadata 字典（不修改原始 chunk_meta）。
        """
        result = dict(chunk_meta)
        p = Path(file_path)

        for field_def in self.schema.metadata_fields:
            value = self._extract_field(field_def, chunk_text, chunk_meta, frontmatter, p)
            if value is None and field_def.required:
                raise ValueError(
                    f"Required metadata field '{field_def.name}' could not be extracted "
                    f"from chunk. File: {file_path}"
                )
            if value is not None:
                result[field_def.name] = self._cast(value, field_def.type)
            elif field_def.default is not None:
                result[field_def.name] = field_def.default

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_field(
        self,
        field_def: MetadataFieldDef,
        chunk_text: str,
        chunk_meta: dict[str, Any],
        frontmatter: dict[str, Any],
        file_path: Path,
    ) -> Any:
        src = field_def.source

        if src == "static":
            return field_def.value

        if src == "filename":
            return file_path.stem

        if src == "filepath":
            return str(file_path)

        if src == "heading_path":
            return chunk_meta.get("heading_path")

        if src == "frontmatter":
            return frontmatter.get(field_def.name)

        if src == "regex":
            if not field_def.pattern:
                return None
            match = re.search(field_def.pattern, chunk_text)
            if match:
                # 返回第一个捕获组，没有捕获组时返回整个匹配
                return match.group(1) if match.lastindex else match.group(0)
            return None

        return None

    @staticmethod
    def _cast(value: Any, target_type: str) -> Any:
        """将提取到的值转换为目标类型。"""
        if value is None:
            return None
        try:
            if target_type == "int":
                return int(value)
            if target_type == "float":
                return float(value)
            if target_type == "bool":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "yes")
            if target_type == "list":
                if isinstance(value, list):
                    return value
                return [value]
            return str(value)
        except (ValueError, TypeError):
            return value  # 转换失败时返回原值
