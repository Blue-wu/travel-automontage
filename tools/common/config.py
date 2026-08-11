"""全局配置管理 — 从环境变量或配置文件加载"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
ASSETS_DB_PATH = DATA_DIR / "assets_db" / "assets.sqlite3"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PIPELINE_DEFS_DIR = PROJECT_ROOT / "pipeline_defs"
SKILLS_DIR = PROJECT_ROOT / "skills"
AGENTS_SKILLS_DIR = PROJECT_ROOT / "agents_skills"


@dataclass
class GeminiConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-1.5-pro"


@dataclass
class CLIPConfig:
    model_name: str = "ViT-L/14"
    device: str = "cpu"  # 或 "cuda"


@dataclass
class DouyinConfig:
    """抖音趋势分析配置"""
    oceanengine_base_url: str = "https://trendinsight.oceanengine.com"
    open_platform_app_id: str = field(default_factory=lambda: os.getenv("DOUYIN_APP_ID", ""))
    open_platform_app_secret: str = field(default_factory=lambda: os.getenv("DOUYIN_APP_SECRET", ""))
    # 人工采集的爆款视频样本路径
    manual_samples_dir: Path = field(default_factory=lambda: DATA_DIR / "samples" / "douyin_viral")


@dataclass
class FFmpegConfig:
    binary: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    # 标准化参数
    target_width: int = 1080
    target_height: int = 1920  # 默认竖屏 9:16
    target_fps: int = 30
    target_codec: str = "libx264"
    target_crf: int = 18
    target_preset: str = "medium"


@dataclass
class Settings:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    douyin: DouyinConfig = field(default_factory=DouyinConfig)
    ffmpeg: FFmpegConfig = field(default_factory=FFmpegConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
