"""视频特征提取 — 基于 VideoMAE 的视频级理解

VideoMAE 是自监督预训练的视频理解模型，
能捕捉时序信息（镜头运动、画面变化节奏），
比单帧 CLIP 更能代表"视频内容"。
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """VideoMAE 视频特征提取器

    用法：
        extractor = VideoFeatureExtractor()
        feat = extractor.extract_shot("video.mp4", 0.0, 5.0)
        # 返回 768 维向量 (base 模型)
    """

    DEFAULT_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
    SMALL_MODEL = "MCG-NJU/videomae-small-finetuned-kinetics"

    def __init__(
        self,
        model_name: str = SMALL_MODEL,
        device: str | None = None,
        num_frames: int = 16,
    ):
        self.model_name = model_name
        self.num_frames = num_frames
        self._model = None
        self._processor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_model(self):
        if self._model is not None:
            return

        logger.info(f"加载 VideoMAE 模型: {self.model_name} (device={self._device})")
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self._processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        self._model = VideoMAEModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        logger.info("VideoMAE 模型加载完成")

    def extract_shot(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[float] | None:
        """提取单个镜头的视频特征

        从镜头中均匀采样 num_frames 帧，送入 VideoMAE，
        取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 作为镜头级特征。

        Args:
            video_path: 视频文件路径
            start_sec: 镜头开始时间
            end_sec: 镜头结束时间

        Returns:
            归一化后的特征向量 (list[float])
        """
        try:
            self._ensure_model()
        except Exception as e:
            logger.warning(f"VideoMAE 加载失败: {e}")
            return None

        frames = self._sample_frames(video_path, start_sec, end_sec)
        if not frames or len(frames) < 8:
            logger.debug(f"采样帧数不足 ({len(frames)})，跳过")
            return None

        try:
            inputs = self._processor(
                list(frames),
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self._device)

            with torch.no_grad():
                outputs = self._model(pixel_values=pixel_values)
                # 优先用 pooler_output（base 及以上模型有）
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feat = outputs.pooler_output.squeeze(0)
                else:
                    # small 模型没有 pooler_output，用 last_hidden_state 全局均值池化
                    last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
                    feat = last_hidden.mean(dim=1).squeeze(0)  # 全局均值

            feat = feat / feat.norm()
            return feat.cpu().tolist()

        except Exception as e:
            logger.warning(f"VideoMAE 特征提取失败: {e}")
            return None

    def _sample_frames(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[np.ndarray]:
        """从视频片段中均匀采样帧

        使用 PyAV 解码，比 ffmpeg 抽帧更快更省 IO。
        """
        import av

        duration = max(0.1, end_sec - start_sec)
        timestamps = [start_sec + duration * i / (self.num_frames - 1) for i in range(self.num_frames)]

        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]

            for ts in timestamps:
                try:
                    container.seek(int(ts * av.time_base), stream=stream)
                    for frame in container.decode(stream):
                        if frame.pts is not None:
                            frame_ts = float(frame.pts * stream.time_base)
                            if frame_ts >= ts - 0.05:
                                img = frame.to_ndarray(format="rgb24")
                                frames.append(img)
                                break
                except Exception:
                    continue

            container.close()
        except Exception as e:
            logger.debug(f"PyAV 采样失败，回退到 ffmpeg: {e}")
            frames = self._sample_frames_ffmpeg(video_path, start_sec, end_sec)

        return frames

    def _sample_frames_ffmpeg(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[np.ndarray]:
        """用 ffmpeg 抽帧的回退方案"""
        from PIL import Image

        duration = max(0.1, end_sec - start_sec)
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = []
            for i in range(self.num_frames):
                ts = start_sec + duration * i / (self.num_frames - 1)
                frame_path = Path(tmpdir) / f"frame_{i:03d}.jpg"
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-ss", f"{ts:.3f}", "-i", video_path,
                            "-frames:v", "1", "-q:v", "3",
                            "-vf", "scale=224:224",
                            str(frame_path), "-y",
                        ],
                        check=True, capture_output=True,
                    )
                    if frame_path.exists():
                        img = np.array(Image.open(frame_path).convert("RGB"))
                        frames.append(img)
                except Exception:
                    continue
            return frames


def aggregate_frame_embeddings(embeddings: list[list[float]], method: str = "mean") -> list[float]:
    """聚合多帧图像嵌入为场景级向量

    Args:
        embeddings: 多帧的 CLIP 嵌入列表
        method: 聚合方法
            - "mean": 均值池化（默认，简单有效）
            - "max": 最大值池化
            - "mean_max": 均值和最大值拼接

    Returns:
        归一化后的聚合向量
    """
    if not embeddings:
        return []

    arr = np.array(embeddings, dtype=np.float32)

    if method == "mean":
        agg = arr.mean(axis=0)
    elif method == "max":
        agg = arr.max(axis=0)
    elif method == "mean_max":
        mean_vec = arr.mean(axis=0)
        max_vec = arr.max(axis=0)
        agg = np.concatenate([mean_vec, max_vec])
    else:
        agg = arr.mean(axis=0)

    agg = agg / (np.linalg.norm(agg) + 1e-8)
    return agg.tolist()
