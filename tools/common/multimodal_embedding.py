"""多模态向量模块 — 基于阿里云 DashScope

使用 qwen3-vl-embedding 模型，将文本、图片、视频转换到同一向量空间，
实现跨模态语义检索（文本搜视频）。

核心模型：
- qwen3-vl-embedding: 文本+图片+视频统一空间，默认2560维，支持多种维度
- qwen2.5-vl-embedding: 旧版模型，2048维

关键特性：文本和视频在同一向量空间，直接用余弦相似度做"文本→视频"检索。
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """多模态向量生成器

    将文本、图片、视频统一编码到同一向量空间。

    用法：
        embedder = MultimodalEmbedder(api_key="sk-xxx")
        text_vec = embedder.embed_text("航拍草原")          # 2560维
        video_vec = embedder.embed_video("clip.mp4", 0, 5)  # 2560维
        # 直接算余弦相似度即可检索
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen3-vl-embedding",
        dim: int = 1024,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.dim = dim
        self._available: bool | None = None
        self._disabled: bool = False  # 连续失败后暂时禁用，避免反复请求

    def is_available(self) -> bool:
        if self._disabled:
            return False
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key)
        return self._available

    def _handle_api_error(self, code: str, message: str) -> None:
        """处理 API 错误，对于欠费/额度问题自动禁用"""
        fatal_codes = {"Arrearage", "AccessDenied", "InvalidApiKey"}
        if any(c in (code or "") for c in fatal_codes):
            logger.warning(f"向量模型被禁用（{code}）：{message}，将回退到 CLIP")
            self._disabled = True
        # 连续失败计数，避免打爆 API
        self._fail_count = getattr(self, "_fail_count", 0) + 1
        if self._fail_count >= 5:
            logger.warning(f"向量模型连续失败 {self._fail_count} 次，暂时禁用，回退到 CLIP")
            self._disabled = True

    def embed_text(self, text: str) -> list[float] | None:
        """文本 → 向量"""
        if not self.is_available():
            return None

        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = self.api_key

        try:
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": text}],
            )
            if resp.status_code != 200:
                logger.error(f"文本向量失败: {resp.code} - {resp.message}")
                self._handle_api_error(resp.code, resp.message)
                return None

            embedding = resp.output["embeddings"][0]["embedding"]
            return self._normalize(embedding)

        except Exception as e:
            logger.error(f"文本向量化异常: {e}")
            return None

    def embed_image(self, image_path: str) -> list[float] | None:
        """图片 → 向量"""
        if not self.is_available():
            return None

        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = self.api_key

        # 检查图片大小（限制 10MB）
        size_mb = Path(image_path).stat().st_size / (1024 * 1024)
        if size_mb > 10:
            logger.warning(f"图片过大 ({size_mb:.1f}MB)，跳过")
            return None

        try:
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{"image": f"file://{image_path}"}],
            )
            if resp.status_code != 200:
                logger.error(f"图片向量失败: {resp.code} - {resp.message}")
                self._handle_api_error(resp.code, resp.message)
                return None

            embedding = resp.output["embeddings"][0]["embedding"]
            return self._normalize(embedding)

        except Exception as e:
            logger.error(f"图片向量化异常: {e}")
            return None

    def embed_video(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_size_mb: float = 50.0,
    ) -> list[float] | None:
        """视频片段 → 向量

        Args:
            video_path: 视频文件路径
            start_sec: 片段开始时间
            end_sec: 片段结束时间（None 表示到视频结尾）
            max_size_mb: 视频最大允许大小（MB）

        Returns:
            归一化的向量 (list[float])
        """
        if not self.is_available():
            return None

        # 准备视频文件（可能需要裁剪或压缩）
        clip_path = self._prepare_video_clip(
            video_path, start_sec, end_sec, max_size_mb
        )
        if clip_path is None:
            return None

        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = self.api_key

        try:
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{"video": f"file://{clip_path}"}],
            )

            if resp.status_code != 200:
                logger.error(f"视频向量失败: {resp.code} - {resp.message}")
                self._handle_api_error(resp.code, resp.message)
                return None

            embedding = resp.output["embeddings"][0]["embedding"]
            return self._normalize(embedding)

        except Exception as e:
            logger.error(f"视频向量化异常: {e}")
            return None
        finally:
            # 清理临时文件
            if clip_path != video_path and os.path.exists(clip_path):
                os.unlink(clip_path)

    def embed_text_and_video(
        self,
        text: str,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> list[float] | None:
        """文本+视频融合向量（融合模式）

        将文本和视频融合为同一个向量，适合需要综合语义的场景。
        """
        if not self.is_available():
            return None

        clip_path = self._prepare_video_clip(video_path, start_sec, end_sec, 50.0)
        if clip_path is None:
            return None

        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = self.api_key

        try:
            resp = MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": text, "video": f"file://{clip_path}"}],
            )

            if resp.status_code != 200:
                logger.error(f"融合向量失败: {resp.code} - {resp.message}")
                self._handle_api_error(resp.code, resp.message)
                return None

            embedding = resp.output["embeddings"][0]["embedding"]
            return self._normalize(embedding)

        except Exception as e:
            logger.error(f"融合向量化异常: {e}")
            return None
        finally:
            if clip_path != video_path and os.path.exists(clip_path):
                os.unlink(clip_path)

    def _prepare_video_clip(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float | None,
        max_size_mb: float,
    ) -> str | None:
        """准备视频片段：裁剪 + 压缩到限制大小以内"""
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        need_clip = start_sec > 0.1 or end_sec is not None
        need_compress = file_size_mb > max_size_mb

        if not need_clip and not need_compress:
            return video_path

        output_path = str(Path(tempfile.mktemp(suffix=".mp4", prefix="me_clip_")))

        # 获取视频时长
        if end_sec is None:
            probe = self._ffprobe(video_path)
            duration = float(probe.get("format", {}).get("duration", 0))
            end_sec = duration

        clip_dur = max(0.1, end_sec - start_sec)

        # 计算目标码率
        target_bitrate = int(max_size_mb * 8 * 1024 * 1024 / clip_dur * 0.9)
        target_bitrate = min(target_bitrate, 2_000_000)  # 最高 2Mbps

        cmd = [
            "ffmpeg",
            "-ss", f"{start_sec:.3f}",
            "-to", f"{end_sec:.3f}",
            "-i", video_path,
            "-c:v", "libx264",
            "-b:v", str(target_bitrate),
            "-vf", "scale=-2:480",  # 降到 480p 够用了
            "-preset", "fast",
            "-an",  # 去掉音频，减小体积
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return output_path
        except Exception as e:
            logger.warning(f"视频片段准备失败: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None

    def _ffprobe(self, video_path: str) -> dict:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", video_path],
                capture_output=True, text=True, check=True,
            )
            import json
            return json.loads(result.stdout)
        except Exception:
            return {}

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
