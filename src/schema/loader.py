"""
SchemaLoader — 从 YAML 文件加载并验证 CollectionSchema。

支持两种使用方式：
    1. 按 collection 名称查找（自动在 schemas/ 目录下查找同名 yaml）
    2. 直接传入 yaml 文件路径

用法：
    schema = SchemaLoader.load("research_articles")
    schema = SchemaLoader.load_file("schemas/my_schema.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .models import CollectionSchema, EmbedderConfig, MetadataFieldDef

# 默认 schemas 目录，相对于项目根（即本文件的上两级）
_DEFAULT_SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"


class SchemaLoader:

    @classmethod
    def load(
        cls,
        collection: str,
        schemas_dir: str | Path | None = None,
    ) -> CollectionSchema:
        """
        按 collection 名称加载 schema。

        Args:
            collection: collection 名称，例如 "research_articles"。
            schemas_dir: schemas 目录路径，默认为项目根目录下的 schemas/。

        Returns:
            CollectionSchema 实例。

        Raises:
            FileNotFoundError: 找不到对应的 yaml 文件。
        """
        base_dir = Path(schemas_dir) if schemas_dir else _DEFAULT_SCHEMAS_DIR
        yaml_path = base_dir / f"{collection}.yaml"

        if not yaml_path.exists():
            # 也尝试 .yml
            yaml_path = base_dir / f"{collection}.yml"

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Schema file not found for collection '{collection}'. "
                f"Expected: {base_dir}/{collection}.yaml"
            )

        return cls.load_file(yaml_path)

    @classmethod
    def load_file(cls, path: str | Path) -> CollectionSchema:
        """
        从指定路径加载 schema yaml 文件。

        Args:
            path: yaml 文件路径。

        Returns:
            CollectionSchema 实例。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        return cls._parse(raw, source=str(path))

    @classmethod
    def load_dict(cls, data: dict[str, Any]) -> CollectionSchema:
        """从字典直接构建 CollectionSchema（用于运行时动态传参）。"""
        return cls._parse(data, source="<dict>")

    # ------------------------------------------------------------------
    # 内部解析
    # ------------------------------------------------------------------

    @classmethod
    def _parse(cls, raw: dict[str, Any], source: str) -> CollectionSchema:
        if "collection" not in raw:
            raise ValueError(f"Schema from {source} must have a 'collection' field.")

        # Embedder
        emb_raw: dict[str, Any] = raw.get("embedder", {})
        if isinstance(emb_raw, str):
            # 简写：embedder: ollama
            emb_raw = {"provider": emb_raw}
        embedder_cfg = EmbedderConfig(
            provider=emb_raw.get("provider", os.environ.get("EMBEDDER_PROVIDER", "ollama")),
            model=emb_raw.get("model", "bge-m3"),
            extra={k: v for k, v in emb_raw.items() if k not in ("provider", "model")},
        )

        # Metadata fields
        fields: list[MetadataFieldDef] = []
        for fdef in raw.get("metadata_fields", []):
            fields.append(cls._parse_field(fdef, source))

        return CollectionSchema(
            collection=raw["collection"],
            chunker=raw.get("chunker", "heading"),
            chunker_config=raw.get("chunker_config", {}),
            embedder=embedder_cfg,
            metadata_fields=fields,
            distance=raw.get("distance", "Cosine"),
        )

    @classmethod
    def _parse_field(cls, fdef: dict[str, Any], source: str) -> MetadataFieldDef:
        if "name" not in fdef:
            raise ValueError(f"Each metadata_field must have a 'name'. In schema: {source}")

        valid_types = {"string", "int", "float", "bool", "list"}
        ftype = fdef.get("type", "string")
        if ftype not in valid_types:
            raise ValueError(
                f"Invalid type '{ftype}' for field '{fdef['name']}'. "
                f"Valid types: {valid_types}"
            )

        valid_sources = {"heading_path", "frontmatter", "regex", "static", "filename", "filepath"}
        fsource = fdef.get("source", "static")
        if fsource not in valid_sources:
            raise ValueError(
                f"Invalid source '{fsource}' for field '{fdef['name']}'. "
                f"Valid sources: {valid_sources}"
            )

        if fsource == "regex" and not fdef.get("pattern"):
            raise ValueError(
                f"Field '{fdef['name']}' has source='regex' but no 'pattern' specified."
            )

        return MetadataFieldDef(
            name=fdef["name"],
            type=ftype,
            source=fsource,
            pattern=fdef.get("pattern"),
            value=fdef.get("value"),
            default=fdef.get("default"),
            required=fdef.get("required", False),
        )
