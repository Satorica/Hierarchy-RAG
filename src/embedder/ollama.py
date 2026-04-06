"""
OllamaEmbedder — 使用本地 Ollama 服务生成向量。

环境变量（优先级高于 schema.yaml config）：
    OLLAMA_HOST       默认 http://localhost:11434
    OLLAMA_EMBED_MODEL 默认 bge-m3

schema.yaml / 代码配置示例：
    embedder:
      provider: ollama
      model: bge-m3
      host: http://localhost:11434
      batch_size: 32   # 每次 API 调用的文本数量

常用模型（需提前 ollama pull）：
    bge-m3              → 1024 维，中英文双语，推荐首选
    nomic-embed-text    → 768 维，英文为主
    mxbai-embed-large   → 1024 维，英文
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .base import BaseEmbedder

# 常见模型的维度映射（避免额外 API 调用）
_KNOWN_DIMENSIONS: dict[str, int] = {
    "bge-m3": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class OllamaEmbedder(BaseEmbedder):
    """调用 Ollama /api/embed 接口生成向量。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        raw_host: str = os.environ.get("OLLAMA_HOST") or self.config.get("host", "http://localhost:11434")
        self.host: str = raw_host.rstrip("/")
        self.model: str = os.environ.get("OLLAMA_EMBED_MODEL") or self.config.get("model", "bge-m3")
        self.batch_size: int = self.config.get("batch_size", 32)
        self._dimension: int | None = None

    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # 批量调用，避免单次请求过大
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._call_ollama(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension

        # 先查已知映射
        for key, dim in _KNOWN_DIMENSIONS.items():
            if key in self.model.lower():
                self._dimension = dim
                return dim

        # 否则发送一条测试请求来探测维度
        sample = self._call_ollama(["hello"])
        self._dimension = len(sample[0])
        return self._dimension

    # ------------------------------------------------------------------

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.host}/api/embed"
        payload = {"model": self.model, "input": texts}

        try:
            resp = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            ) from e

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}\nResponse: {resp.text}") from e

        data = resp.json()

        # Ollama /api/embed 返回 {"embeddings": [[...]]}
        if "embeddings" in data:
            return data["embeddings"]

        # 兼容旧版 /api/embeddings 单条格式
        if "embedding" in data:
            return [data["embedding"]]

        raise RuntimeError(f"Unexpected Ollama response format: {list(data.keys())}")
