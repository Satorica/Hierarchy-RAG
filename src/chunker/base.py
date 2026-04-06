"""
BaseChunker — 用户自定义 chunker 的核心协议。

用法：
    from src.chunker.base import BaseChunker, Chunk

    class MyChunker(BaseChunker):
        def chunk(self, text: str, base_metadata: dict) -> list[Chunk]:
            # 你的分块逻辑
            ...

    # 在 schema.yaml 中注册：
    # chunker: my_package.my_module.MyChunker
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """一个分块单元，包含文本内容和携带的 metadata。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id[:8]}, len={len(self.text)}, text={preview!r}...)"


class BaseChunker(ABC):
    """
    所有 chunker 必须继承此类并实现 `chunk` 方法。

    Args:
        config: 来自 schema.yaml chunker_config 节点的字典，可以为空。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """
        将文本切分为若干 Chunk。

        Args:
            text: 原始文档文本（纯 Markdown 或纯文本）。
            base_metadata: 文档级别的基础 metadata（如 source_file、collection 等），
                           chunker 可以在每个 chunk 的 metadata 中合并或覆盖这些字段。

        Returns:
            Chunk 列表，顺序保持原文档顺序。
        """
        ...

    def get_name(self) -> str:
        """返回 chunker 的标识名，用于日志。"""
        return self.__class__.__name__
