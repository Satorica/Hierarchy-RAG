"""
BasePDFParser — PDF 解析器的统一抽象接口。

用法（自定义解析器）：
    from src.parser.base import BasePDFParser, ParseResult

    class MyParser(BasePDFParser):
        def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    """PDF 解析结果。"""

    markdown_text: str               # 提取的完整 Markdown 文本
    source_path: str                 # 原始 PDF 路径
    images: list[str] = field(default_factory=list)    # 提取的图片文件路径列表
    metadata: dict[str, Any] = field(default_factory=dict)  # 解析器额外返回的 meta（页数等）
    parser_name: str = ""

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


class BasePDFParser(ABC):
    """
    所有 PDF 解析器必须继承此类并实现 `parse` 方法。

    Args:
        config: 解析器配置字典。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
        """
        解析 PDF 文件为 Markdown 文本。

        Args:
            pdf_path: PDF 文件绝对路径。
            output_dir: 图片等附属文件的输出目录（可选，解析器自行决定是否使用）。

        Returns:
            ParseResult 实例。
        """
        ...

    def get_name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _ensure_output_dir(pdf_path: str, output_dir: str | None) -> Path:
        """辅助：确保输出目录存在并返回 Path。"""
        if output_dir:
            out = Path(output_dir)
        else:
            out = Path(pdf_path).parent / (Path(pdf_path).stem + "_parsed")
        out.mkdir(parents=True, exist_ok=True)
        return out
