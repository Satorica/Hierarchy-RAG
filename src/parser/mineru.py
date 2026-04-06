"""
MineruParser — 调用本地 MinerU（conda 环境）解析 PDF。

依赖：
    - conda 环境 mineru（含 mineru==3.x）
    - 串行执行，不支持并发（Mac Mini 资源约束）

环境变量：
    MINERU_CONDA_ENV   默认 mineru
    MINERU_METHOD      默认 txt（可选 auto / ocr）
    MINERU_BACKEND     默认 pipeline
    MINERU_LANG        默认 en

schema.yaml / 代码配置示例：
    parser:
      provider: mineru
      conda_env: mineru
      method: txt
      backend: pipeline
      lang: en
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import BasePDFParser, ParseResult


class MineruParser(BasePDFParser):
    """通过 conda run 调用 MinerU CLI 解析 PDF。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.conda_env: str = os.environ.get("MINERU_CONDA_ENV") or self.config.get("conda_env", "mineru")
        self.method: str = os.environ.get("MINERU_METHOD") or self.config.get("method", "txt")
        self.backend: str = os.environ.get("MINERU_BACKEND") or self.config.get("backend", "pipeline")
        self.lang: str = os.environ.get("MINERU_LANG") or self.config.get("lang", "en")

    def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
        pdf_path = str(Path(pdf_path).resolve())
        out_dir = self._ensure_output_dir(pdf_path, output_dir)

        cmd = [
            "conda", "run", "-n", self.conda_env, "--no-capture-output",
            "mineru",
            "-p", pdf_path,
            "-o", str(out_dir),
            "-m", self.method,
            "--backend", self.backend,
            "--lang", self.lang,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 分钟超时
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"MinerU parsing timed out for {pdf_path}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "conda not found. Make sure conda is installed and in PATH."
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU failed (returncode={result.returncode}).\n"
                f"STDERR: {result.stderr[-2000:]}"
            )

        # 查找输出 markdown 文件
        md_files = glob.glob(str(out_dir / "**" / "*.md"), recursive=True)
        if not md_files:
            raise RuntimeError(
                f"MinerU ran successfully but no .md file found in {out_dir}"
            )

        # 取最大的 md 文件（通常只有一个）
        md_path = max(md_files, key=os.path.getsize)
        with open(md_path, encoding="utf-8") as f:
            markdown_text = f.read()

        # 收集图片文件
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        images = [
            str(p) for p in out_dir.rglob("*")
            if p.suffix.lower() in image_extensions
        ]

        return ParseResult(
            markdown_text=markdown_text,
            source_path=pdf_path,
            images=images,
            metadata={
                "md_path": md_path,
                "output_dir": str(out_dir),
                "method": self.method,
            },
            parser_name="MineruParser",
        )
