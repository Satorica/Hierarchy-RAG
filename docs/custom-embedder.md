# Custom Embedder Guide

本文档说明如何接入任意 embedding 模型，包括本地模型、第三方 API 以及多模态 embedding。

---

## 接口约定

实现一个 embedder 只需满足：

```
输入：list[str]（文本列表）
输出：list[list[float]]（对应的向量列表，顺序一致）
```

框架还需要知道向量维度（用于 Qdrant 建集合），通过 `dimension` 属性提供。

---

## 内置 Embedder 快速使用

### Ollama（默认）

```yaml
# schema.yaml
embedder:
  provider: ollama
  model: bge-m3          # 需提前运行: ollama pull bge-m3
```

```bash
# 环境变量覆盖
OLLAMA_HOST=http://192.168.1.100:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

常用 Ollama 模型：

| 模型 | 维度 | 特点 |
|------|------|------|
| `bge-m3` | 1024 | 中英双语，首选 |
| `nomic-embed-text` | 768 | 英文，速度快 |
| `mxbai-embed-large` | 1024 | 英文，精度高 |
| `all-minilm` | 384 | 轻量，适合资源受限环境 |

### OpenAI

```yaml
embedder:
  provider: openai
  model: text-embedding-3-small    # 或 text-embedding-3-large
```

```bash
OPENAI_API_KEY=sk-...
# 兼容其他 OpenAI 格式 API（如 Azure、本地 vLLM）：
OPENAI_BASE_URL=https://your-endpoint.openai.azure.com/
```

---

## 自定义 Embedder

继承 `BaseEmbedder`，实现两个内容：
1. `embed_texts(texts)` — 核心方法
2. `dimension` 属性 — 返回向量维度整数

```python
from src.embedder.base import BaseEmbedder
from typing import Any


class MyEmbedder(BaseEmbedder):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # config 来自 schema.yaml 的 embedder 节点或代码传入

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 返回与 texts 等长的向量列表
        ...

    @property
    def dimension(self) -> int:
        return 768  # 你的模型的向量维度
```

### 示例 1：使用 sentence-transformers 本地模型

```python
# my_embedders/st_embedder.py
from sentence_transformers import SentenceTransformer
from src.embedder.base import BaseEmbedder
from typing import Any


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    使用 sentence-transformers 加载任意 HuggingFace 模型。
    pip install sentence-transformers

    schema.yaml 配置：
        embedder:
          provider: my_embedders.st_embedder.SentenceTransformerEmbedder
          model: BAAI/bge-large-zh-v1.5
          batch_size: 64
          device: cuda          # 或 cpu / mps
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        model_name = self.config.get("model", "BAAI/bge-large-zh-v1.5")
        device = self.config.get("device", "cpu")
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = self.config.get("batch_size", 64)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,   # 余弦相似度需要归一化
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
```

在 `schema.yaml` 中引用：

```yaml
embedder:
  provider: my_embedders.st_embedder.SentenceTransformerEmbedder
  model: BAAI/bge-large-zh-v1.5
  device: mps        # Apple Silicon
  batch_size: 32
```

### 示例 2：调用本地 vLLM / LiteLLM 兼容接口

```python
# my_embedders/litellm_embedder.py
import requests
from src.embedder.base import BaseEmbedder
from typing import Any


class LiteLLMEmbedder(BaseEmbedder):
    """
    调用任何 OpenAI-compatible /v1/embeddings 端点。
    兼容：vLLM、LiteLLM proxy、Xinference、FastChat 等。

    schema.yaml 配置：
        embedder:
          provider: my_embedders.litellm_embedder.LiteLLMEmbedder
          base_url: http://localhost:8000
          model: BAAI/bge-m3
          api_key: none         # 本地部署通常不需要
          dimension: 1024
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_url = self.config.get("base_url", "http://localhost:8000").rstrip("/")
        self._model = self.config.get("model", "BAAI/bge-m3")
        self._api_key = self.config.get("api_key", "none")
        self._dim = self.config.get("dimension", 1024)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self._base_url}/v1/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        # 按 index 排序确保顺序正确
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    @property
    def dimension(self) -> int:
        return self._dim
```

### 示例 3：Cohere Embed v3

```python
# my_embedders/cohere_embedder.py
import cohere
from src.embedder.base import BaseEmbedder
from typing import Any
import os


class CohereEmbedder(BaseEmbedder):
    """
    使用 Cohere Embed v3 API。
    pip install cohere

    schema.yaml 配置：
        embedder:
          provider: my_embedders.cohere_embedder.CohereEmbedder
          model: embed-multilingual-v3.0
          input_type: search_document    # ingest 时用 search_document
                                         # query 时用 search_query
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        api_key = os.environ.get("COHERE_API_KEY") or self.config.get("api_key")
        self._client = cohere.Client(api_key)
        self._model = self.config.get("model", "embed-multilingual-v3.0")
        self._input_type = self.config.get("input_type", "search_document")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=self._input_type,
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        # embed-multilingual-v3.0 → 1024 维
        return 1024
```

---

## 在 MCP Server 中配置 Embedder

Embedder 配置**唯一的正确位置是 `schema.yaml`**。

`query_collection` tool 不接受 embedder 参数——查询时框架自动从对应 collection 的
schema.yaml 读取 embedder，与 ingest 时完全一致。这是强制约束，不可绕过。

原因：ingest 和 query 必须使用相同的 embedding 模型。不同模型的向量空间互不兼容，
混用会让检索结果变成随机噪声，且不会有任何报错提示。

### 不同 collection 使用不同 embedder

每个 collection 的 schema.yaml 独立配置，互不影响：

```yaml
# schemas/research_articles.yaml
embedder:
  provider: my_embedders.st_embedder.SentenceTransformerEmbedder
  model: BAAI/bge-large-zh-v1.5
  device: mps

# schemas/personal_notes.yaml
embedder:
  provider: ollama
  model: nomic-embed-text
```

查询时只需指定 collection 名，embedder 自动对应：

```json
{"tool": "query_collection", "arguments": {"query": "...", "collection": "research_articles"}}
{"tool": "query_collection", "arguments": {"query": "...", "collection": "personal_notes"}}
```

两次查询会分别用 `SentenceTransformerEmbedder` 和 `OllamaEmbedder` 处理查询向量，与各自 ingest 时一致。

### 修改 embedder 后需要重新 ingest

如果修改了 schema.yaml 中的 embedder（换了模型），**必须重新 ingest 所有文档**，否则存量向量与新模型的查询向量来自不同空间，检索完全失效。

```bash
# 删除旧 collection 重建
# manage_collection: action=delete, collection=research_articles
# 然后重新 ingest 所有文档
```
