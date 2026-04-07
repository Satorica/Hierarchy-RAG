"""
MineruCloudParser — 调用 MinerU Cloud 精准解析 API 解析 PDF。

使用精准解析 API（/api/v4/extract/task），支持：
- 最大 200MB / 600 页
- 异步提交 + 动态轮询任务状态（带进度打印）
- 从结果 zip 中提取 full.md

环境变量：
    MINERU_API_TOKEN   必填，从 https://mineru.net/apiManage 申请

schema.yaml / 代码配置示例：
    parser:
      provider: mineru_cloud
      model_version: vlm          # pipeline（默认）/ vlm（推荐）
      language: en
      enable_formula: true
      enable_table: true
      poll_interval: 5            # 轮询间隔秒数，默认 5
      poll_timeout: 600           # 最长等待秒数，默认 600

Token 配置（只放环境变量，不要写进代码或 yaml）：
    export MINERU_API_TOKEN=your_token_here
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from .base import BasePDFParser, ParseResult

_BASE_URL = "https://mineru.net/api/v4"


class MineruCloudParser(BasePDFParser):
    """
    通过 MinerU Cloud 精准解析 API 解析 PDF。

    流程：
        1. 申请批量上传链接（/file-urls/batch）
        2. PUT 上传本地 PDF 到 OSS
        3. 轮询批量任务状态（/extract-results/batch/{batch_id}）
        4. 下载 zip → 解压 full.md → 返回 ParseResult
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        token = os.environ.get("MINERU_API_TOKEN") or self.config.get("api_token")
        if not token:
            raise ValueError(
                "MINERU_API_TOKEN environment variable is required for MineruCloudParser.\n"
                "Set it with: export MINERU_API_TOKEN=your_token_here\n"
                "Apply for a token at: https://mineru.net/apiManage"
            )
        self._token = token
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        self.model_version: str = self.config.get("model_version", "vlm")
        self.language: str = self.config.get("language", "en")
        self.enable_formula: bool = self.config.get("enable_formula", True)
        self.enable_table: bool = self.config.get("enable_table", True)
        self.poll_interval: int = self.config.get("poll_interval", 5)
        self.poll_timeout: int = self.config.get("poll_timeout", 600)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str, output_dir: str | None = None) -> ParseResult:
        """
        上传 PDF 到 MinerU Cloud，等待解析完成，返回 Markdown 文本。

        Args:
            pdf_path: 本地 PDF 文件路径。
            output_dir: 可选，zip 解压目录（默认在 pdf 同级目录下创建 _cloud_parsed/）。

        Returns:
            ParseResult，其中 markdown_text 为 full.md 内容。
        """
        pdf_path = str(Path(pdf_path).resolve())
        pdf_name = Path(pdf_path).name
        out_dir = self._ensure_output_dir(pdf_path, output_dir)

        print(f"[MineruCloud] Uploading: {pdf_name}")

        # Step 1: 申请上传链接 + batch_id
        batch_id, upload_url = self._request_upload_url(pdf_name)
        print(f"[MineruCloud] batch_id={batch_id}")

        # Step 2: PUT 上传文件到 OSS
        self._upload_file(pdf_path, upload_url)
        print(f"[MineruCloud] Upload complete. Waiting for parse...")

        # Step 3: 轮询任务状态
        full_zip_url = self._poll_batch(batch_id, pdf_name)
        print(f"[MineruCloud] Parse done. Downloading result zip...")

        # Step 4: 下载 zip，提取 full.md
        markdown_text, images = self._download_and_extract(full_zip_url, out_dir)
        print(f"[MineruCloud] Markdown length={len(markdown_text)}, images={len(images)}")

        return ParseResult(
            markdown_text=markdown_text,
            source_path=pdf_path,
            images=images,
            metadata={
                "batch_id": batch_id,
                "model_version": self.model_version,
                "output_dir": str(out_dir),
            },
            parser_name="MineruCloudParser",
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _request_upload_url(self, filename: str) -> tuple[str, str]:
        """申请批量上传链接，返回 (batch_id, upload_url)。"""
        url = f"{_BASE_URL}/file-urls/batch"
        payload = {
            "files": [{"name": filename}],
            "model_version": self.model_version,
            "language": self.language,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
        }
        resp = requests.post(url, headers=self._headers, json=payload, timeout=30)
        self._raise_for_api_error(resp, "Request upload URL")

        data = resp.json()["data"]
        batch_id: str = data["batch_id"]
        upload_url: str = data["file_urls"][0]
        return batch_id, upload_url

    def _upload_file(self, pdf_path: str, upload_url: str) -> None:
        """PUT 上传本地文件到 OSS。"""
        with open(pdf_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=120)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"File upload failed: HTTP {resp.status_code}\n{resp.text[:500]}"
            )

    def _poll_batch(self, batch_id: str, filename: str) -> str:
        """
        轮询批量任务，直到目标文件状态为 done。
        动态打印进度。返回 full_zip_url。
        """
        url = f"{_BASE_URL}/extract-results/batch/{batch_id}"
        deadline = time.time() + self.poll_timeout
        state_label = {
            "waiting-file": "waiting for upload",
            "pending": "queued",
            "running": "parsing",
            "converting": "converting",
            "done": "done",
            "failed": "failed",
        }

        while time.time() < deadline:
            resp = requests.get(url, headers=self._headers, timeout=30)
            self._raise_for_api_error(resp, "Poll batch status")

            results: list[dict] = resp.json()["data"]["extract_result"]
            # 找到对应文件的记录（batch 里只有一个文件）
            file_result = next(
                (r for r in results if r.get("file_name") == filename),
                results[0] if results else None,
            )
            if not file_result:
                raise RuntimeError(f"No result found for file '{filename}' in batch {batch_id}")

            state: str = file_result["state"]
            elapsed = int(self.poll_timeout - (deadline - time.time()))

            if state == "running":
                prog = file_result.get("extract_progress", {})
                extracted = prog.get("extracted_pages", "?")
                total = prog.get("total_pages", "?")
                print(f"[MineruCloud] [{elapsed}s] parsing... {extracted}/{total} pages")
            elif state == "done":
                return file_result["full_zip_url"]
            elif state == "failed":
                err = file_result.get("err_msg", "unknown error")
                raise RuntimeError(f"[MineruCloud] Parse failed: {err}")
            else:
                label = state_label.get(state, state)
                print(f"[MineruCloud] [{elapsed}s] {label}...")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"MinerU Cloud parse timed out after {self.poll_timeout}s. "
            f"batch_id={batch_id}"
        )

    def _download_and_extract(
        self, zip_url: str, out_dir: Path
    ) -> tuple[str, list[str]]:
        """
        下载结果 zip，解压，提取 full.md 文本和图片路径列表。

        Returns:
            (markdown_text, image_paths)
        """
        resp = requests.get(zip_url, timeout=120)
        resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        zf.extractall(out_dir)

        # 找 full.md（在 zip 内可能位于子目录）
        md_candidates = [n for n in zf.namelist() if n.endswith("full.md")]
        if not md_candidates:
            # 兼容旧版 zip 结构（有时叫 *.md 不叫 full.md）
            md_candidates = [n for n in zf.namelist() if n.endswith(".md")]
        if not md_candidates:
            raise RuntimeError(
                f"No .md file found in result zip. Contents: {zf.namelist()[:20]}"
            )

        # 取最大的 md 文件
        md_name = max(md_candidates, key=lambda n: len(zf.read(n)))
        markdown_text = zf.read(md_name).decode("utf-8")

        # 收集图片
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        images = [
            str(out_dir / name)
            for name in zf.namelist()
            if Path(name).suffix.lower() in image_exts
        ]

        zf.close()
        return markdown_text, images

    @staticmethod
    def _raise_for_api_error(resp: requests.Response, context: str) -> None:
        """检查 HTTP 状态和 API code，失败时抛出可读错误。"""
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"[MineruCloud] {context} HTTP error: {e}") from e

        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(
                f"[MineruCloud] {context} API error: "
                f"code={body.get('code')} msg={body.get('msg')}"
            )
