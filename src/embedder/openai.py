"""
OpenAIEmbedder — 使用 OpenAI Embeddings API 生成向量。

环境变量：
    OPENAI_API_KEY    必填
    OPENAI_BASE_URL   可选，用于兼容 OpenAI-compatible API（如 Azure、第三方代理）

schema.yaml / 代码配置示例：
    embedder:
      provider: openai
      model: text-embedding-3-small   # 或 text-embedding-3-large
      batch_size: 100
      dimensions: 1536   # text-embedding-3-* 系列支持自定义维度（可选）

常用模型：
    text-embedding-3-small  → 1536 维（可降维），性价比最高
    text-embedding-3-large  → 3072 维，精度最高
    text-embedding-ada-002  → 1536 维，旧款
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseEmbedder

_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """调用 OpenAI Embeddings API 生成向量。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model: str = self.config.get("model", "text-embedding-3-small")
        self.batch_size: int = self.config.get("batch_size", 100)
        self._custom_dimensions: int | None = self.config.get("dimensions")

        # 延迟导入，避免未安装 openai 时报错
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY", self.config.get("api_key"))
        base_url = os.environ.get("OPENAI_BASE_URL", self.config.get("base_url"))

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAIEmbedder."
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = _openai.OpenAI(**kwargs)

    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self._custom_dimensions:
                kwargs["dimensions"] = self._custom_dimensions

            response = self._client.embeddings.create(**kwargs)
            # 按 index 排序（OpenAI 保证顺序，但以防万一）
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings

    @property
    def dimension(self) -> int:
        if self._custom_dimensions:
            return self._custom_dimensions
        for key, dim in _KNOWN_DIMENSIONS.items():
            if key in self.model:
                return dim
        # 未知模型：探测
        sample = self.embed_texts(["hello"])
        return len(sample[0])
