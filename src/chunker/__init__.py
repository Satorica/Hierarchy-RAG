"""
Chunker 模块。

内置 chunker 可通过短名称直接使用：
    heading   → HeadingChunker
    fixed     → FixedSizeChunker
    paragraph → ParagraphChunker

也可以指定完整的 Python 模块路径来加载自定义 chunker：
    chunker: my_package.my_module.MyChunker
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import BaseChunker, Chunk
from .fixed import FixedSizeChunker
from .heading import HeadingChunker
from .paragraph import ParagraphChunker

# 内置 chunker 注册表：短名称 → 类
BUILTIN_CHUNKERS: dict[str, type[BaseChunker]] = {
    "heading": HeadingChunker,
    "fixed": FixedSizeChunker,
    "paragraph": ParagraphChunker,
}


def load_chunker(name: str, config: dict[str, Any] | None = None) -> BaseChunker:
    """
    根据名称加载 chunker 实例。

    Args:
        name: 内置短名称（如 "heading"）或完整模块路径（如 "my_pkg.my_mod.MyChunker"）。
        config: 传递给 chunker 的配置字典。

    Returns:
        初始化好的 BaseChunker 实例。

    Raises:
        ValueError: 名称无法解析时抛出。
    """
    if name in BUILTIN_CHUNKERS:
        return BUILTIN_CHUNKERS[name](config=config)

    # 尝试按 "module.path.ClassName" 动态加载
    if "." in name:
        module_path, class_name = name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not issubclass(cls, BaseChunker):
                raise TypeError(f"{name} must be a subclass of BaseChunker")
            return cls(config=config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load chunker '{name}': {e}") from e

    available = ", ".join(BUILTIN_CHUNKERS.keys())
    raise ValueError(
        f"Unknown chunker '{name}'. Built-in options: {available}. "
        "Or use a full module path like 'my_pkg.my_mod.MyChunker'."
    )


__all__ = [
    "BaseChunker",
    "Chunk",
    "HeadingChunker",
    "FixedSizeChunker",
    "ParagraphChunker",
    "load_chunker",
    "BUILTIN_CHUNKERS",
]
