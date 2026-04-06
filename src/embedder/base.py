"""
BaseEmbedder — Embedding provider 的统一抽象接口。

用法（自定义 provider）：
    from src.embedder.base import BaseEmbedder

    class MyEmbedder(BaseEmbedder):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            ...

    # 在 schema.yaml 或环境变量中配置：
    # EMBEDDER_PROVIDER=my_pkg.my_mod.MyEmbedder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedder(ABC):
    """
    所有 embedding provider 必须继承此类。

    Args:
        config: provider 配置字典（模型名、host、api_key 等）。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量将文本转为向量。

        Args:
            texts: 待 embed 的字符串列表。

        Returns:
            每个文本对应的浮点向量列表，顺序与输入一致。
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """便捷方法：单条文本 embed。"""
        return self.embed_texts([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度，用于 Qdrant 建集合时指定。"""
        ...

    def get_name(self) -> str:
        return self.__class__.__name__
