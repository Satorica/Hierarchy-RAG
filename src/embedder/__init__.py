"""
Embedder 模块。

内置 provider 短名称：
    ollama  → OllamaEmbedder（默认）
    openai  → OpenAIEmbedder

也可通过完整模块路径加载自定义 provider：
    my_pkg.my_mod.MyEmbedder
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from .base import BaseEmbedder
from .ollama import OllamaEmbedder
from .openai import OpenAIEmbedder

BUILTIN_EMBEDDERS: dict[str, type[BaseEmbedder]] = {
    "ollama": OllamaEmbedder,
    "openai": OpenAIEmbedder,
}


def load_embedder(provider: str | None = None, config: dict[str, Any] | None = None) -> BaseEmbedder:
    """
    加载 embedder 实例。

    Args:
        provider: 短名称（"ollama"/"openai"）或完整模块路径。
                  若为 None，从环境变量 EMBEDDER_PROVIDER 读取，默认 "ollama"。
        config: provider 配置字典。

    Returns:
        初始化好的 BaseEmbedder 实例。
    """
    if provider is None:
        provider = os.environ.get("EMBEDDER_PROVIDER", "ollama")

    if provider in BUILTIN_EMBEDDERS:
        return BUILTIN_EMBEDDERS[provider](config=config)

    if "." in provider:
        module_path, class_name = provider.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not issubclass(cls, BaseEmbedder):
                raise TypeError(f"{provider} must be a subclass of BaseEmbedder")
            return cls(config=config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load embedder '{provider}': {e}") from e

    available = ", ".join(BUILTIN_EMBEDDERS.keys())
    raise ValueError(
        f"Unknown embedder provider '{provider}'. Built-in options: {available}."
    )


__all__ = [
    "BaseEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "load_embedder",
    "BUILTIN_EMBEDDERS",
]
