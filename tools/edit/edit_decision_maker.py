"""剪辑决策器 — 生最终剪辑指令：入点出点、转场、字幕、特效

Agent 在 edit_decision 阶段调用此工具，
同时读取 skills/travel-pacing.md 获取节奏规范。
"""

from __future__ import annotations

import logging
from typing import Any

from tools.common.models import (
    BgmConfig,
    EditDecision,
    OutputFormat,
    TimelineItem,
    VoiceoverConfig,
)

logger = logging.getLogger(__name__)


class EditDecisionMaker:
    """剪辑决策器"""

    def decide(
        self,
        storyboard: list[dict[str, Any]],
        output_format: dict[str, Any] | None = None,
        bgm: dict[str, Any] | None = None,
    ) -> EditDecision:
        """生成剪辑决策

        Args:
            storyboard: 分镜列表
            output_format: 输出格式
            bgm: BGM 配置

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
            in_sec = float(matched_clip.get("start_sec", 0))
            out_sec = float(matched_clip.get("end_sec", in_sec + 3))

            # 分镜时长
            duration = item.get("duration_sec", out_sec - in_sec)
            if "script_scene" in item:
                script_scene = item["script_scene"]
                if isinstance(script_scene, dict):
                    duration = float(script_scene.get("duration_sec", duration))

            # 确保时长不超过素材实际长度
            duration = min(duration, out_sec - in_sec)
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

        decision = EditDecision(
            timeline=timeline,
            bgm=bgm_config,
            voiceover=None,
            output_format=fmt,
            total_duration_sec=total_duration,
        )

        logger.info(f"剪辑决策完成: {len(timeline)} 个片段, 总时长 {total_duration:.1f}s")
        return decision

    def _suggest_effects(self, storyboard_item: dict, clip: dict) -> list[dict[str, Any]]:
        """为片段建议特效"""
        effects = []

        # 检查是否为静态画面（需要 Ken Burns 效果避免幻灯片风险）
        motion_tags = clip.get("motion_tags", [])
        visual_tags = clip.get("visual_tags", [])

        is_static = any("静态" in tag for tag in motion_tags) or not motion_tags
        is_landscape = any("风景" in tag or "全景" in tag or "航拍" in tag for tag in visual_tags)

        if is_static and is_landscape:
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
        """默认字幕样式"""
        return {
            "font_size": 48,
            "position": "bottom",
            "color": "#FFFFFF",
            "stroke": "#000000",
            "stroke_width": 2,
        }
