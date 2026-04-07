"""
Parser 模块。

内置 PDF 解析器：
    mineru_cloud → MineruCloudParser（调用 MinerU Cloud API，推荐，需 MINERU_API_TOKEN）
    mineru       → MineruParser（调用本地 MinerU conda 环境）
    pymupdf      → PyMuPDFParser（纯 Python，无需 conda，适合简单 PDF）

也可通过完整模块路径加载自定义解析器：
    my_pkg.my_mod.MyParser
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import BasePDFParser, ParseResult
from .mineru import MineruParser
from .mineru_cloud import MineruCloudParser
from .pymupdf import PyMuPDFParser

BUILTIN_PARSERS: dict[str, type[BasePDFParser]] = {
    "mineru_cloud": MineruCloudParser,
    "mineru": MineruParser,
    "pymupdf": PyMuPDFParser,
}


def load_parser(provider: str = "mineru", config: dict[str, Any] | None = None) -> BasePDFParser:
    """
    加载 PDF 解析器实例。

    Args:
        provider: 短名称（"mineru"/"pymupdf"）或完整模块路径。
        config: 传递给解析器的配置字典。

    Returns:
        初始化好的 BasePDFParser 实例。
    """
    if provider in BUILTIN_PARSERS:
        return BUILTIN_PARSERS[provider](config=config)

    if "." in provider:
        module_path, class_name = provider.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not issubclass(cls, BasePDFParser):
                raise TypeError(f"{provider} must be a subclass of BasePDFParser")
            return cls(config=config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load parser '{provider}': {e}") from e

    available = ", ".join(BUILTIN_PARSERS.keys())
    raise ValueError(
        f"Unknown parser provider '{provider}'. Built-in options: {available}."
    )


__all__ = [
    "BasePDFParser",
    "ParseResult",
    "MineruCloudParser",
    "MineruParser",
    "PyMuPDFParser",
    "load_parser",
    "BUILTIN_PARSERS",
]
