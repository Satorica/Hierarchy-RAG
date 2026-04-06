"""
Schema 数据模型 — 描述一个 collection 的配置结构。

对应 schema.yaml 的完整字段定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class MetadataFieldDef:
    """
    单个 metadata 字段的定义。

    source 取值：
        heading_path  — 从当前 chunk 的标题路径自动提取（chunker 填写）
        frontmatter   — 从 Markdown YAML frontmatter 中按 name 取值
        regex         — 在 chunk 文本中用正则提取第一个捕获组
        static        — 固定值（value 字段）
        filename      — 取文件名（不含扩展名）
        filepath      — 取文件完整路径
    """

    name: str
    type: Literal["string", "int", "float", "bool", "list"] = "string"
    source: Literal["heading_path", "frontmatter", "regex", "static", "filename", "filepath"] = "static"
    pattern: str | None = None    # source=regex 时使用
    value: Any = None             # source=static 时使用
    default: Any = None           # 提取失败时的默认值
    required: bool = False        # 若为 True 且提取失败则抛出错误


@dataclass
class EmbedderConfig:
    """Embedder 配置。"""
    provider: str = "ollama"
    model: str = "bge-m3"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionSchema:
    """
    一个 collection 的完整 schema 配置。
    对应 schemas/<collection_name>.yaml。
    """

    collection: str                          # collection 名称（即 Qdrant collection name）
    chunker: str = "heading"                 # chunker 短名称或完整模块路径
    chunker_config: dict[str, Any] = field(default_factory=dict)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    metadata_fields: list[MetadataFieldDef] = field(default_factory=list)

    # 可选：指定 Qdrant distance metric
    distance: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
