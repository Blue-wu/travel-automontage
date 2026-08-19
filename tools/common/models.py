"""Pydantic 数据模型 — 所有阶段共享的数据结构"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Scene(BaseModel):
    """视频场景"""
    start: str = Field(description="起始时间码 HH:MM:SS")
    end: str = Field(description="结束时间码 HH:MM:SS")
    start_sec: float = 0.0
    end_sec: float = 0.0
    summary: str = ""
    visual_tags: list[str] = Field(default_factory=list)
    scene_category: str = "other"
    motion_tags: list[str] = Field(default_factory=list)
    audio_tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    video_embedding: list[float] | None = None
    quality: float = 0.0
    people_count: int = 0
    dominant_colors: list[str] = Field(default_factory=list)


class AssetMetadata(BaseModel):
    duration: float = 0.0
    has_audio: bool = False
    quality_score: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    tags: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.now)


class Asset(BaseModel):
    """入库后的素材资产"""
    asset_id: str
    source_path: str
    normalized_path: str = ""
    destination: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    metadata: AssetMetadata = Field(default_factory=AssetMetadata)


class HotTopic(BaseModel):
    keyword: str
    search_volume: int = 0
    content_volume: int = 0
    growth_rate: float = 0.0


class ViralPattern(BaseModel):
    pattern_name: str
    description: str
    frequency: float = 0.0
    example_videos: list[str] = Field(default_factory=list)


class HookFormula(BaseModel):
    hook_type: str  # suspense, contrast, resonance, number, question, shock
    template: str
    effectiveness: float = 0.0
    example_text: str = ""


class PacingRule(BaseModel):
    rule: str
    avg_shot_duration: float = 0.0
    description: str = ""


class MusicTrend(BaseModel):
    bgm_name: str
    usage_count: int = 0
    mood: str = ""


class ContentTypeSuggestion(BaseModel):
    """内容类型建议"""
    type: str  # "guide"攻略型, "emotional"情感型, "informational"信息型, "vlog"日常Vlog型
    name: str
    description: str
    popularity: float = 0.0
    storytelling_tips: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)


class TrendReport(BaseModel):
    """抖音趋势分析报告"""
    niche: str
    destination: str | None = None
    analyzed_at: datetime = Field(default_factory=datetime.now)
    hot_topics: list[HotTopic] = Field(default_factory=list)
    viral_patterns: list[ViralPattern] = Field(default_factory=list)
    hook_formulas: list[HookFormula] = Field(default_factory=list)
    pacing_rules: list[PacingRule] = Field(default_factory=list)
    music_trends: list[MusicTrend] = Field(default_factory=list)
    content_type_suggestions: list[ContentTypeSuggestion] = Field(default_factory=list)
    recommended_content_type: ContentTypeSuggestion | None = None


class CandidateClip(BaseModel):
    """检索到的候选素材片段"""
    asset_id: str
    source_path: str
    start_sec: float
    end_sec: float
    score: float
    scene_summary: str = ""
    visual_tags: list[str] = Field(default_factory=list)
    scene_category: str = "other"
    quality: float = 0.0


class SceneCategory(BaseModel):
    """素材场景分类统计"""
    category: str  # mountain/lake/grassland/forest/road/sunset/night/sky/architecture/food
    display_name: str
    count: int = 0
    clips: list[CandidateClip] = Field(default_factory=list)


class CandidateClips(BaseModel):
    query: str
    destination: str
    total_found: int = 0
    clips: list[CandidateClip] = Field(default_factory=list)
    categories: list[SceneCategory] = Field(default_factory=list)
    total_available_duration: float = 0.0
    estimated_output_duration: float = 0.0


class ScriptScene(BaseModel):
    """剧本中的分镜"""
    order: int
    duration_sec: float
    clip_description: str
    narration: str = ""
    subtitle: str = ""
    transition: str = "cut"
    mood: str = ""
    visual_keywords: list[str] = Field(default_factory=list)
    shot_type: str = "medium"


class Hook(BaseModel):
    type: str
    text: str
    duration_sec: float = 3.0


class BgmSuggestion(BaseModel):
    name: str = ""
    mood: str = ""
    bpm: int = 0


class Script(BaseModel):
    """创意剧本"""
    title: str
    destination: str = ""
    duration: float = 30.0
    hook: Hook | None = None
    bgm_suggestion: BgmSuggestion | None = None
    voiceover: str = ""
    scenes: list[ScriptScene] = Field(default_factory=list)
    cta: str = ""


class TimelineItem(BaseModel):
    """时间线上的单个片段"""
    order: int
    asset_id: str
    source_path: str
    in_sec: float
    out_sec: float
    duration_sec: float
    transition_in: str = "cut"
    transition_duration: float = 0.5
    subtitle: str = ""
    subtitle_style: dict[str, Any] = Field(default_factory=dict)
    effects: list[dict[str, Any]] = Field(default_factory=list)
    narration_text: str = ""


class BgmConfig(BaseModel):
    path: str = ""
    volume: float = 0.4
    fade_in_sec: float = 0.5
    fade_out_sec: float = 1.0


class VoiceoverConfig(BaseModel):
    path: str = ""
    volume: float = 1.0


class OutputFormat(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    codec: str = "h264"
    preset: str = "medium"


class EditDecision(BaseModel):
    """剪辑决策"""
    timeline: list[TimelineItem] = Field(default_factory=list)
    bgm: BgmConfig | None = None
    voiceover: VoiceoverConfig | None = None
    output_format: OutputFormat = Field(default_factory=OutputFormat)
    total_duration_sec: float = 0.0


class QACheck(BaseModel):
    """单项质检结果"""
    name: str
    passed: bool
    score: float = 0.0
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class QAReview(BaseModel):
    """质检报告"""
    video_path: str
    reviewed_at: datetime = Field(default_factory=datetime.now)
    passed: bool = False
    overall_score: float = 0.0
    checks: list[QACheck] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CreativeBrief(BaseModel):
    """创作 brief — 文案的「活人感」原料

    trip_context 是唯一真正不可自动生成的输入。
    见 skills/travel-copywriting.md §6 / §9。
    """
    destination: str = ""
    # ★ 创作者提供的真实行程背景，1-3 句。没有它文案必然退化成套话
    trip_context: str = ""
    # 口吻：克制 / 文艺 / 沙雕 / 干货 / 知心朋友
    persona: str = "克制、不煽情、像跟朋友讲事"
    audience: str = ""
    # 禁用词，会与 skill 里的禁用清单合并
    avoid: list[str] = Field(default_factory=list)


class TripFacts(BaseModel):
    """从素材元数据自动重建的行程事实骨架（skills/travel-copywriting.md §9）"""
    date_range: str = ""
    days: int = 0
    locations: list[str] = Field(default_factory=list)
    # 每天拍了什么类型的场景
    daily_scene_types: dict[str, list[str]] = Field(default_factory=dict)
    # 最早/最晚的拍摄时刻 —— "十点天还亮着"这类句子的来源
    earliest_shot_time: str = ""
    latest_shot_time: str = ""
    # 停留最久 / 素材最密集的地点
    densest_location: str = ""
    total_clips: int = 0

    def to_prompt_block(self) -> str:
        lines = []
        if self.date_range:
            lines.append(f"- 拍摄日期：{self.date_range}（共 {self.days} 天）")
        if self.locations:
            lines.append(f"- 路线：{' → '.join(self.locations)}")
        if self.latest_shot_time:
            lines.append(f"- 每天最晚拍到：{self.latest_shot_time}")
        if self.earliest_shot_time:
            lines.append(f"- 每天最早拍摄：{self.earliest_shot_time}")
        if self.densest_location:
            lines.append(f"- 素材最集中的地点：{self.densest_location}（停留最久）")
        for day, types in list(self.daily_scene_types.items())[:8]:
            lines.append(f"- {day}：{'、'.join(types)}")
        return "\n".join(lines)
