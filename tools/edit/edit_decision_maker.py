"""剪辑决策器 — 生最终剪辑指令：入点出点、转场、字幕、特效

Agent 在 edit_decision 阶段调用此工具，
同时读取 skills/travel-pacing.md 获取节奏规范。
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from tools.common.models import (
    BgmConfig,
    EditDecision,
    OutputFormat,
    TimelineItem,
    VoiceoverConfig,
)

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频总时长"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


class EditDecisionMaker:
    """剪辑决策器"""

    def decide(
        self,
        storyboard: list[dict[str, Any]],
        output_format: dict[str, Any] | None = None,
        bgm: dict[str, Any] | None = None,
        voiceover: dict[str, Any] | None = None,
    ) -> EditDecision:
        """生成剪辑决策

        Args:
            storyboard: 分镜列表
            output_format: 输出格式
            bgm: BGM 配置
            voiceover: 旁白配音配置

        Returns:
            EditDecision: 剪辑决策
        """
        logger.info(f"生成剪辑决策: {len(storyboard)} 个分镜")

        timeline = []
        total_duration = 0.0

        for item in storyboard:
            matched_clip = item.get("matched_clip")
            if not matched_clip:
                continue

            # 提取素材信息
            asset_id = matched_clip.get("asset_id", "")
            source_path = matched_clip.get("source_path", "")
            clip_start = float(matched_clip.get("start_sec", 0))
            clip_end = float(matched_clip.get("end_sec", clip_start + 3))
            clip_mid = (clip_start + clip_end) / 2

            # 分镜时长（剧本要求的时长）
            target_duration = item.get("duration_sec", clip_end - clip_start)
            if "script_scene" in item:
                script_scene = item["script_scene"]
                if isinstance(script_scene, dict):
                    target_duration = float(script_scene.get("duration_sec", target_duration))

            # 获取视频总时长，以场景为中心向前后扩展
            video_duration = _get_video_duration(source_path)
            if video_duration > 0 and target_duration > (clip_end - clip_start):
                # 以场景中点为中心，扩展到目标时长
                half = target_duration / 2
                in_sec = max(0.1, clip_mid - half)
                out_sec = min(video_duration, clip_mid + half)
                # 如果靠边了，往另一边挪
                if in_sec < 0.1:
                    out_sec = min(video_duration, target_duration)
                    in_sec = 0.1
                if out_sec > video_duration:
                    in_sec = max(0.1, video_duration - target_duration)
                    out_sec = video_duration
                duration = out_sec - in_sec
            else:
                # 视频太短或获取失败，用原片段
                in_sec = clip_start
                out_sec = clip_end
                duration = min(target_duration, out_sec - in_sec)
                out_sec = in_sec + duration

            # 转场
            transition = item.get("transition", "cut")
            transition_duration = 0.5 if transition != "cut" else 0.0

            # 字幕
            subtitle = ""
            subtitle_style = {}
            if "script_scene" in item:
                script_scene = item["script_scene"]
                if isinstance(script_scene, dict):
                    subtitle = script_scene.get("subtitle", "")
                    narration = script_scene.get("narration", "")
                    # 如果有旁白但无字幕，用旁白的前几个字做字幕
                    if not subtitle and narration:
                        subtitle = narration[:8]

            # 效果（Ken Burns 等）
            effects = self._suggest_effects(item, matched_clip)

            timeline_item = TimelineItem(
                order=len(timeline) + 1,
                asset_id=asset_id,
                source_path=source_path,
                in_sec=in_sec,
                out_sec=out_sec,
                duration_sec=duration,
                transition_in=transition,
                transition_duration=transition_duration,
                subtitle=subtitle,
                subtitle_style=self._default_subtitle_style(),
                effects=effects,
                narration_text=item.get("script_scene", {}).get("narration", "") if isinstance(item.get("script_scene"), dict) else "",
                # 画面原料透传给 copywriting stage 做错位抓手
                visual_summary=matched_clip.get("scene_summary", "") or "",
                # subjects 优先：具体物件才是错位手法的抓手，visual_tags 可能是类别同义词
                visual_tags=list(
                    matched_clip.get("subjects")
                    or matched_clip.get("visual_tags")
                    or []
                ),
            )

            timeline.append(timeline_item)
            total_duration += duration

        # 输出格式
        fmt = OutputFormat(**(output_format or {}))

        # BGM 配置
        bgm_config = None
        if bgm:
            bgm_config = BgmConfig(
                path=bgm.get("path", ""),
                volume=bgm.get("volume", 0.4),
                fade_in_sec=bgm.get("fade_in_sec", 0.5),
                fade_out_sec=bgm.get("fade_out_sec", 1.0),
            )

        vo_config = None
        if voiceover and voiceover.get("path"):
            vo_config = VoiceoverConfig(
                path=voiceover["path"],
                volume=voiceover.get("volume", 1.0),
            )

        decision = EditDecision(
            timeline=timeline,
            bgm=bgm_config,
            voiceover=vo_config,
            output_format=fmt,
            total_duration_sec=total_duration,
        )

        logger.info(f"剪辑决策完成: {len(timeline)} 个片段, 总时长 {total_duration:.1f}s")
        return decision

    def _suggest_effects(self, storyboard_item: dict, clip: dict) -> list[dict[str, Any]]:
        """为片段建议特效"""
        effects = []

        # 所有镜头强制加 Ken Burns，避免幻灯片风险
        effects.append({
            "type": "ken_burns",
            "intensity": 0.3,
        })

        # 高潮段加 vignette
        mood = storyboard_item.get("script_scene", {})
        if isinstance(mood, dict):
            scene_mood = mood.get("mood", "")
            if scene_mood in ("epic", "shock"):
                effects.append({
                    "type": "vignette",
                    "intensity": 0.4,
                })

        return effects

    def _default_subtitle_style(self) -> dict[str, Any]:
        """默认字幕样式（横屏 1080p 基准 font_size；4K scale=2 → 约 165px 实际渲染）"""
        return {
            "font_size": 72,
            "position": "bottom",
            "color": "#FFFFFF",
            "stroke": "#000000",
            "stroke_width": 4,
        }
