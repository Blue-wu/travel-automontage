"""素材入库模块 — 把"杂乱素材"变成"可被语义检索的资产库"

流程：扫描目录 → 转码 → 上传模型API（带路径缓存）→ 逐段分析 → 向量化 → 写入素材库
参考 Montai 的做法：转码为模型可处理格式，上传到 Gemini File API（路径键缓存避免重复上传），
逐段分析每个视频生成场景摘要。
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Iterator

from tools.common.asset_store import AssetStore
from tools.common.config import CACHE_DIR, get_settings
from tools.common.model_client import ModelClient
from tools.common.models import Asset, AssetMetadata, Scene

logger = logging.getLogger(__name__)

# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}


class TravelAssetIngestor:
    """旅行素材入库器

    扫描素材目录 → 转码 → 上传模型API → 逐段分析 → 写入素材库

    用法：
        ingestor = TravelAssetIngestor()
        ingestor.ingest("~/Travel/Fuji2024")
    """

    def __init__(
        self,
        asset_store: AssetStore | None = None,
        model_client: ModelClient | None = None,
    ):
        self.asset_store = asset_store or AssetStore()
        self.model_client = model_client or ModelClient()
        self.settings = get_settings()

    def ingest(self, footage_dir: str, skip_existing: bool = True) -> list[Asset]:
        """入库整个目录的素材

        Args:
            footage_dir: 素材目录路径
            skip_existing: 是否跳过已入库的素材（基于路径去重）

        Returns:
            入库成功的 Asset 列表
        """
        footage_path = Path(footage_dir).expanduser()
        if not footage_path.exists():
            raise FileNotFoundError(f"素材目录不存在: {footage_path}")

        ingested = []
        for video_path in self._scan_videos(footage_path):
            try:
                # 检查是否已入库
                if skip_existing:
                    existing = self.asset_store.get_by_path(str(video_path))
                    if existing is not None:
                        logger.info(f"跳过已入库: {video_path.name}")
                        ingested.append(existing)
                        continue

                asset = self._ingest_single(video_path)
                if asset:
                    self.asset_store.save(asset)
                    ingested.append(asset)
                    logger.info(f"入库成功: {video_path.name} ({len(asset.scenes)} 场景)")
                else:
                    logger.warning(f"入库失败: {video_path.name}")
            except Exception as e:
                logger.error(f"入库异常 {video_path.name}: {e}")

        logger.info(f"入库完成: {len(ingested)}/{len(list(self._scan_videos(footage_path)))} 成功")
        return ingested

    def _ingest_single(self, video_path: Path) -> Asset | None:
        """入库单个视频文件"""
        asset_id = self._generate_asset_id(str(video_path))

        # Step 1: 转码为统一格式
        normalized_path = self._transcode(video_path)

        # Step 2: 逐段分析（上传到多模态模型 + 路径缓存）
        analysis = self.model_client.analyze_video_scenes(str(normalized_path))

        # Step 3: 构建场景列表 + 向量化
        scenes = self._build_scenes(analysis, str(normalized_path))

        # Step 4: 构建 Asset
        metadata = self._build_metadata(analysis, str(video_path))

        return Asset(
            asset_id=asset_id,
            source_path=str(video_path),
            normalized_path=str(normalized_path),
            destination=analysis.get("destination", ""),
            scenes=scenes,
            metadata=metadata,
        )

    def _scan_videos(self, directory: Path) -> Iterator[Path]:
        """扫描目录中的视频文件"""
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path

    def _generate_asset_id(self, video_path: str) -> str:
        """基于文件路径 + 大小 + 修改时间生成唯一 ID"""
        p = Path(video_path)
        stat = p.stat()
        raw = f"{p.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _transcode(self, video_path: Path) -> Path:
        """转码为统一格式（H.264, 保持原分辨率, 30fps）

        如果已经是标准格式则直接返回原路径。
        """
        cache_dir = CACHE_DIR / "normalized"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_path = cache_dir / f"{video_path.stem}_normalized.mp4"

        # 如果缓存已存在且比源文件新，直接用缓存
        if output_path.exists() and output_path.stat().st_mtime > video_path.stat().st_mtime:
            return output_path

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-r", "30",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            "-y", str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"转码失败 {video_path}: {e.stderr[:200]}")
            # 转码失败则用原文件
            return video_path

    def _build_scenes(self, analysis: dict, normalized_path: str) -> list[Scene]:
        """从分析结果构建场景列表，并尝试向量化"""
        scenes_data = analysis.get("scenes", [])
        scenes = []

        for s in scenes_data:
            scene = Scene(
                start=s.get("start", "00:00:00"),
                end=s.get("end", "00:00:10"),
                start_sec=float(s.get("start_sec", 0)),
                end_sec=float(s.get("end_sec", 10)),
                summary=s.get("summary", ""),
                visual_tags=s.get("visual_tags", []),
                motion_tags=s.get("motion_tags", []),
                audio_tags=s.get("audio_tags", []),
                quality=float(s.get("quality", 0.7)),
            )

            # 尝试 CLIP 向量化
            try:
                mid_sec = (scene.start_sec + scene.end_sec) / 2
                embedding = self.model_client.embed_video_frame(normalized_path, mid_sec)
                if embedding:
                    scene.embedding = embedding
            except Exception as e:
                logger.debug(f"向量化失败（非致命）: {e}")

            scenes.append(scene)

        return scenes

    def _build_metadata(self, analysis: dict, source_path: str) -> AssetMetadata:
        """构建素材元数据"""
        meta = analysis.get("metadata", {})
        probe = self.model_client.ffprobe(source_path)

        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = float(probe.get("format", {}).get("duration", meta.get("duration", 0)))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        return AssetMetadata(
            duration=duration,
            has_audio=audio_stream is not None,
            quality_score=float(analysis.get("quality_score", 0.7)),
            resolution=f"{width}x{height}",
            fps=fps,
            tags=analysis.get("tags", []),
        )
