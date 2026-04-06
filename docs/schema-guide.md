# Schema YAML Guide

`schemas/<collection_name>.yaml` 是每个 collection 的完整配置文件，控制：

- 使用哪种 chunker 及其参数
- 使用哪种 embedding 模型
- 每个 chunk 应该携带哪些 metadata 字段、如何提取

---

## 完整字段说明

```yaml
# ── 必填 ──────────────────────────────────────────────────────
collection: my_collection           # Qdrant collection 名称，全局唯一

# ── Chunker ───────────────────────────────────────────────────
chunker: heading                    # 内置：heading / fixed / paragraph
                                    # 自定义：my_pkg.my_module.MyChunker
chunker_config:                     # 传给 chunker 构造函数的 config dict
  max_chunk_size: 1500              # chunker 自定义参数，见各 chunker 文档

# ── Embedder ──────────────────────────────────────────────────
embedder:
  provider: ollama                  # 内置：ollama / openai
                                    # 自定义：my_pkg.my_module.MyEmbedder
  model: bge-m3                     # 模型名称
  # 其他 key-value 原样传给 embedder 的 config dict
  batch_size: 32
  device: mps

# ── Qdrant 向量配置 ────────────────────────────────────────────
distance: Cosine                    # Cosine（默认）/ Dot / Euclid

# ── Metadata 字段定义 ─────────────────────────────────────────
metadata_fields:
  - name: my_field                  # 字段名（写入 Qdrant payload 的 key）
    type: string                    # 类型：string / int / float / bool / list
    source: static                  # 来源：见下方"source 类型"章节
    value: "some_value"             # source=static 时的固定值
    default: null                   # 提取失败时的默认值（null 表示不写入）
    required: false                 # true 时提取失败会抛出错误中止 ingest
```

---

## source 类型详解

### `static` — 固定值

所有 chunk 都写入同一个固定值，适合标记 collection 来源、版本、领域等。

```yaml
metadata_fields:
  - name: domain
    type: string
    source: static
    value: "biomedical"

  - name: schema_version
    type: int
    source: static
    value: 2
```

**Qdrant 中每个 chunk 的 payload**：
```json
{"domain": "biomedical", "schema_version": 2, ...}
```

查询时按 domain 过滤：
```json
{"tool": "query_collection", "arguments": {"filters": {"domain": "biomedical"}}}
```

---

### `filename` — 文件名（不含扩展名）

```yaml
metadata_fields:
  - name: paper_id
    type: string
    source: filename
```

文件 `/tmp/arxiv_2401.12345_autoplc.pdf` → `paper_id: "arxiv_2401.12345_autoplc"`

---

### `filepath` — 完整文件路径

```yaml
metadata_fields:
  - name: source_path
    type: string
    source: filepath
```

---

### `frontmatter` — 从 Markdown YAML frontmatter 提取

适合有 frontmatter 的 Markdown 文件（Obsidian 笔记、Hugo 博客等）。

字段名 `name` 必须与 frontmatter 中的 key 完全一致。

```markdown
---
title: "深度学习入门"
tags: ["AI", "教程"]
created: "2026-03-01"
author: "张三"
difficulty: 3
---

正文内容...
```

```yaml
metadata_fields:
  - name: title
    type: string
    source: frontmatter
    default: ""

  - name: tags
    type: list
    source: frontmatter
    default: []

  - name: created
    type: string
    source: frontmatter
    default: null

  - name: difficulty
    type: int
    source: frontmatter
    default: 1
```

**注意**：没有 frontmatter 的文档，这些字段取 `default` 值。

---

### `regex` — 正则表达式提取

在每个 chunk 的文本内容中用正则搜索，取**第一个捕获组**。适合从文档内容中提取结构化信息。

```yaml
metadata_fields:
  # 提取图片编号，如 "Figure 3" → "3"
  - name: figure_id
    type: string
    source: regex
    pattern: "(?:Figure|Fig\\.?)\\s+(\\d+)"
    default: null

  # 提取表格编号
  - name: table_id
    type: string
    source: regex
    pattern: "Table\\s+(\\d+)"
    default: null

  # 提取章节编号，如 "3.2 Method" → "3.2"
  - name: section_number
    type: string
    source: regex
    pattern: "^(\\d+(?:\\.\\d+)*)\\s"
    default: null

  # 提取 DOI
  - name: doi
    type: string
    source: regex
    pattern: "10\\.\\d{4,}/[\\S]+"
    default: null

  # 提取年份（第一个出现的4位年份）
  - name: year
    type: int
    source: regex
    pattern: "(20\\d{2}|19\\d{2})"
    default: null
```

**注意**：
- 必须包含至少一个括号捕获组 `(...)`，否则返回整个匹配串
- 只取第一个匹配结果
- 正则作用于 chunk 文本，每个 chunk 独立提取

---

### `heading_path` — 标题路径（仅 HeadingChunker 有效）

由 `HeadingChunker` 在分块时自动写入 `chunk.metadata`，`source: heading_path` 直接读取这个值。

```yaml
chunker: heading
chunker_config:
  inherit_parent_headings: true

metadata_fields:
  - name: heading_path     # "Introduction > Background > Prior Work"
    type: string
    source: heading_path

  - name: section_title    # 只取最后一级标题
    type: string
    source: heading_path   # 注意：此字段实际读取 chunk.metadata["heading_path"]
                           # heading 本身由 chunker 写入 chunk.metadata["heading"]
```

HeadingChunker 自动在 `chunk.metadata` 中写入：
- `heading_path`: `"Introduction > Background"` （完整路径）
- `heading`: `"Background"` （当前节标题）
- `heading_level`: `2` （标题层级深度）

这三个字段不需要在 `metadata_fields` 中声明，chunker 会自动携带。

---

## 完整示例集

### 学术论文 collection

```yaml
collection: research_articles
chunker: heading
chunker_config:
  max_chunk_size: 1500
  min_chunk_size: 80
  inherit_parent_headings: true

embedder:
  provider: ollama
  model: bge-m3

distance: Cosine

metadata_fields:
  - name: paper_id
    type: string
    source: filename

  - name: domain
    type: string
    source: static
    value: "computer_science"

  - name: figure_id
    type: string
    source: regex
    pattern: "(?:Figure|Fig\\.?)\\s+(\\d+)"
    default: null

  - name: table_id
    type: string
    source: regex
    pattern: "Table\\s+(\\d+)"
    default: null

  - name: tags
    type: list
    source: frontmatter
    default: []
```

### Obsidian 个人笔记 collection

```yaml
collection: personal_notes
chunker: paragraph
chunker_config:
  min_paragraph_size: 60
  max_chunk_size: 1000
  merge_short: true

embedder:
  provider: ollama
  model: bge-m3

metadata_fields:
  - name: title
    type: string
    source: frontmatter
    default: ""

  - name: tags
    type: list
    source: frontmatter
    default: []

  - name: created
    type: string
    source: frontmatter
    default: null

  - name: source_url
    type: string
    source: frontmatter
    default: null

  - name: status
    type: string
    source: frontmatter    # frontmatter 中的 status: "draft" / "published"
    default: "draft"
```

### 产品文档 collection（固定字段 + 版本追踪）

```yaml
collection: product_docs
chunker: fixed
chunker_config:
  chunk_size: 600
  overlap: 80
  split_on_newline: true

embedder:
  provider: ollama
  model: nomic-embed-text

metadata_fields:
  - name: product
    type: string
    source: static
    value: "my_product_v2"

  - name: doc_version
    type: string
    source: static
    value: "2.1.0"

  - name: language
    type: string
    source: frontmatter
    default: "zh"

  - name: category
    type: string
    source: frontmatter    # frontmatter: category: "API Reference"
    default: "general"
```

### 自定义物理量标注（科研数据场景）

```yaml
collection: experiment_logs
chunker: paragraph

embedder:
  provider: my_embedders.st_embedder.SentenceTransformerEmbedder
  model: allenai/scibert_scivocab_uncased
  dimension: 768

metadata_fields:
  - name: temperature_k
    type: float
    source: regex
    pattern: "(\\d+(?:\\.\\d+)?)\\s*K"           # 提取温度，如 "298.15 K"
    default: null

  - name: pressure_mpa
    type: float
    source: regex
    pattern: "(\\d+(?:\\.\\d+)?)\\s*MPa"          # 提取压力
    default: null

  - name: experiment_id
    type: string
    source: frontmatter
    required: true                                  # 实验日志必须有 ID

  - name: instrument
    type: string
    source: frontmatter
    default: "unknown"

  - name: lab
    type: string
    source: static
    value: "ChemLab-A"
```

---

## 查询时用 metadata 过滤

ingest 时写入的所有 metadata 字段都可以在查询时用来精准过滤：

```json
{
  "tool": "query_collection",
  "arguments": {
    "query": "实验误差分析",
    "collection": "experiment_logs",
    "top_k": 5,
    "filters": {
      "lab": "ChemLab-A",
      "instrument": "GC-MS"
    }
  }
}
```

---

## 常见问题

**Q：`source: frontmatter` 提取不到值时怎么办？**

A：确保 `default` 字段不为 `null`，或者把 `required` 设为 `false`（默认）。没有 frontmatter 的文档所有 `frontmatter` 字段都会取 `default` 值。

**Q：多个字段都想从正则提取，但 pattern 会互相干扰？**

A：不会干扰。每个字段的正则独立作用于同一 chunk 文本，互相不影响。

**Q：我想在 metadata 里存一个列表，如 `["AI", "NLP"]`？**

A：把 `type` 设为 `list`，`source` 用 `frontmatter` 或 `static`（`value: ["AI", "NLP"]`）。`regex` source 目前只能提取单个字符串，若要提取列表需要自定义 chunker 在 metadata 中直接写入。

**Q：schema.yaml 改了之后，旧的 chunk 会自动更新吗？**

A：不会。metadata 在 ingest 时一次性写入。修改 schema 后需要重新 ingest 文件（`overwrite: true` 默认开启，会先删除旧数据再写入）。
