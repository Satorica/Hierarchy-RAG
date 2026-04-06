"""
PyMuPDFParser — 使用 PyMuPDF (fitz) 解析 PDF，无需 MinerU 环境。

适合：纯文字 PDF，不含复杂公式/表格的场景。
对于学术论文等复杂布局，推荐使用 MineruParser。

依赖：
    pip install pymupdf

环境变量/配置：
    无特殊环境变量，直接在 schema.yaml 的 parser 节点配置。

schema.yaml 配置示例：
    parser:
      provider: pymupdf
      extract_images: false   # 是否提取图片（默认 false）
      image_min_size: 100     # 忽略小于此像素的图片
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BasePDFParser, ParseResult


class PyMuPDFParser(BasePDFParser):
    """使用 PyMuPDF 将 PDF 转换为纯文本 Markdown。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.extract_images: bool = self.config.get("extract_images", False)
        self.image_min_size: int = self.config.get("image_min_size", 100)

    def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PyMuPDFParser. "
                "Install it with: pip install pymupdf"
            ) from e

        pdf_path = str(Path(pdf_path).resolve())
        doc = fitz.open(pdf_path)

        pages_text: list[str] = []
        images: list[str] = []

        out_dir = self._ensure_output_dir(pdf_path, output_dir) if self.extract_images else None

        for page_num, page in enumerate(doc):
            # 提取文字（保留基本块结构）
            blocks = page.get_text("blocks")
            page_lines: list[str] = []

            for block in sorted(blocks, key=lambda b: (b[1], b[0])):  # 按 y, x 排序
                block_text = block[4].strip()
                if block_text:
                    page_lines.append(block_text)

            pages_text.append("\n\n".join(page_lines))

            # 可选图片提取
            if self.extract_images and out_dir:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < self.image_min_size or pix.height < self.image_min_size:
                        continue
                    img_path = out_dir / f"page{page_num + 1}_img{img_index + 1}.png"
                    if pix.n >= 5:  # CMYK 转 RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(img_path))
                    images.append(str(img_path))

        full_text = "\n\n---\n\n".join(pages_text)
        page_count = len(doc)
        doc.close()

        return ParseResult(
            markdown_text=full_text,
            source_path=pdf_path,
            images=images,
            metadata={"page_count": page_count},
            parser_name="PyMuPDFParser",
        )
