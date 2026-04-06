# Custom Chunker & PDF Parser Guide

本文档说明如何自定义 PDF 解析和分块逻辑，对接 Hierarchy-RAG 的 embedding 和存储流程。

---

## 核心原则

框架对中间过程**没有任何约束**。你的解析器和分块器可以用任何库、任何算法，唯一的输出约定是：

```
最终产出一个 list[Chunk]
```

每个 `Chunk` 只需要两个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 用于 embedding 的文本内容（也可以是图片 caption、表格的文字描述等） |
| `metadata` | `dict` | 任意键值对，原样写入 Qdrant payload，查询时可按这些字段过滤 |

`Chunk` 来自 `src.chunker.base`，直接 import 即可：

```python
from src.chunker.base import Chunk
```

---

## 路径一：完全自定义（推荐给有特殊需求的用户）

不继承任何框架类。自己解析 PDF、自己分块，最后调用 `pipeline.ingest_chunks()` 直接写入。

### 示例：用 pdfplumber 解析 + 自定义规则分块

```python
# my_pipeline.py
import pdfplumber
from src.chunker.base import Chunk
from src.ingestor import IngestPipeline
from src.schema import SchemaLoader


def parse_and_chunk(pdf_path: str) -> list[Chunk]:
    """完全自定义的解析 + 分块逻辑，与框架无关。"""
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # 你自己的分块规则：这里按段落切
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para_idx, para in enumerate(paragraphs):
                chunks.append(Chunk(
                    text=para,
                    metadata={
                        "page": page_num,
                        "para_index": para_idx,
                        "source_file": pdf_path,  # 建议加上，用于 overwrite 定位
                    }
                ))

    return chunks


# 接入框架
schema = SchemaLoader.load("research_articles")
pipeline = IngestPipeline(schema)

chunks = parse_and_chunk("/path/to/paper.pdf")
result = pipeline.ingest_chunks(chunks, source_label="/path/to/paper.pdf")
print(result.to_dict())
```

### 示例：处理多模态内容（图片 + 文字混合）

```python
# 用 VLM 为图片生成 caption，和正文 chunk 一起 ingest
chunks = []

# 正文 chunk
chunks.append(Chunk(
    text="3.2 Experimental Results\n\nWe compared three methods...",
    metadata={"section": "3.2", "content_type": "text", "source_file": "paper.pdf"}
))

# 图片 caption chunk（可以用 BLIP2/GPT-4V 生成描述）
chunks.append(Chunk(
    text="Figure 3: Comparison of accuracy across three methods. "
         "Method A achieves 94.2% on dataset X.",
    metadata={"figure_id": "fig3", "content_type": "figure_caption", "source_file": "paper.pdf"}
))

# 表格内容转为自然语言
chunks.append(Chunk(
    text="Table 1: Method A: 94.2%, Method B: 91.5%, Method C: 88.3%",
    metadata={"table_id": "table1", "content_type": "table", "source_file": "paper.pdf"}
))

result = pipeline.ingest_chunks(chunks, source_label="paper.pdf")
```

---

## 路径二：继承 BaseChunker（适合想复用框架加载机制的用户）

如果你的分块器需要通过 `schema.yaml` 配置、或者想在多个 collection 中复用，可以继承 `BaseChunker`。

```python
# my_chunkers/sentence_chunker.py
import re
from src.chunker.base import BaseChunker, Chunk
from typing import Any


class SentenceChunker(BaseChunker):
    """
    按句子切割文本。
    schema.yaml 中配置：
        chunker: my_chunkers.sentence_chunker.SentenceChunker
        chunker_config:
          min_sentence_len: 30
          max_sentences_per_chunk: 5
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.min_len = self.config.get("min_sentence_len", 30)
        self.max_sents = self.config.get("max_sentences_per_chunk", 5)

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        # 按中英文句末标点切割
        sentence_end = re.compile(r"(?<=[。！？.!?])\s+")
        sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]

        # 过滤过短的句子
        sentences = [s for s in sentences if len(s) >= self.min_len]

        chunks = []
        for i in range(0, len(sentences), self.max_sents):
            group = sentences[i : i + self.max_sents]
            chunks.append(Chunk(
                text=" ".join(group),
                metadata={
                    **base_metadata,
                    "sentence_start": i,
                    "sentence_count": len(group),
                }
            ))

        return chunks
```

在 `schema.yaml` 中注册：

```yaml
chunker: my_chunkers.sentence_chunker.SentenceChunker
chunker_config:
  min_sentence_len: 30
  max_sentences_per_chunk: 5
```

---

## 路径三：继承 BasePDFParser（适合替换 PDF 解析引擎）

如果你想让框架自动在 `.pdf` 文件上调用你的解析器（走 `pipeline.ingest()` 而不是手动 `ingest_chunks`），继承 `BasePDFParser`：

```python
# my_parsers/marker_parser.py
from src.parser.base import BasePDFParser, ParseResult
from typing import Any


class MarkerParser(BasePDFParser):
    """
    使用 marker-pdf 库解析 PDF。
    pip install marker-pdf

    环境变量/配置：
        PDF_PARSER=my_parsers.marker_parser.MarkerParser
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.batch_multiplier = self.config.get("batch_multiplier", 2)

    def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        models = load_all_models()
        full_text, images, out_meta = convert_single_pdf(
            pdf_path,
            models,
            batch_multiplier=self.batch_multiplier,
        )

        return ParseResult(
            markdown_text=full_text,
            source_path=pdf_path,
            images=list(images.keys()),    # marker 返回图片路径字典
            metadata=out_meta,
            parser_name="MarkerParser",
        )
```

然后通过环境变量启用：

```bash
PDF_PARSER=my_parsers.marker_parser.MarkerParser python src/mcp_server/main.py --stdio
```

或在调用 `IngestPipeline` 时指定：

```python
pipeline = IngestPipeline(schema, pdf_parser="my_parsers.marker_parser.MarkerParser")
result = pipeline.ingest("/path/to/paper.pdf")
```

---

## 内置 Chunker 参数速查

### HeadingChunker（默认）

```yaml
chunker: heading
chunker_config:
  max_chunk_size: 1500       # 单节最大字符数，超出时按段落二次切割
  min_chunk_size: 50         # 节内容低于此字符数时跳过
  inherit_parent_headings: true  # metadata 中记录完整标题路径 "A > B > C"
  include_heading_in_text: true  # chunk text 中包含标题行本身
```

### FixedSizeChunker

```yaml
chunker: fixed
chunker_config:
  chunk_size: 800     # 每个 chunk 目标字符数
  overlap: 100        # 相邻 chunk 重叠字符数（防止语义断裂）
  split_on_newline: true  # 优先在换行处对齐，避免切断句子
```

### ParagraphChunker

```yaml
chunker: paragraph
chunker_config:
  min_paragraph_size: 80   # 短于此字符的段落与下一段合并
  max_chunk_size: 1200     # 超出此字符的段落按句子再次切割
  merge_short: true
```

---

## 决策树

```
我有自己的 PDF 解析 / 分块代码
    └─ 是 → 直接产出 list[Chunk]，调用 pipeline.ingest_chunks()
               不需要继承任何类

我想通过 schema.yaml 配置分块策略，复用框架加载
    └─ 是 → 继承 BaseChunker，实现 chunk() 方法

我想让框架自动在 .pdf 文件上调用我的解析器
    └─ 是 → 继承 BasePDFParser，实现 parse() 方法
               通过 PDF_PARSER 环境变量或 IngestPipeline(pdf_parser=...) 启用
```
