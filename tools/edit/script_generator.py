"""剧本生成器 — 结合趋势分析 + 素材库生成创意剧本

Agent 在 script_generation 阶段调用此工具，
同时读取 skills/travel-storytelling.md 获取叙事规范。
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from tools.common.models import (
    BgmSuggestion,
    CandidateClips,
    Hook,
    Script,
    ScriptScene,
    TrendReport,
)

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """旅行 Vlog 剧本生成器"""

    # 黄金时间轴模板（30秒）
    GOLDEN_TIMELINE_30S = [
        {"phase": "hook", "start": 0, "end": 3, "description": "钩子 + 最具冲击力画面"},
        {"phase": "intro", "start": 3, "end": 8, "description": "第一转场，引入目的地"},
        {"phase": "develop", "start": 8, "end": 15, "description": "街头美食/人文特写"},
        {"phase": "transit", "start": 15, "end": 22, "description": "交通转移，在路上"},
        {"phase": "climax", "start": 22, "end": 28, "description": "高潮景观 + 情绪抒发"},
        {"phase": "ending", "start": 28, "end": 30, "description": "收尾 + CTA"},
    ]

    # 黄金时间轴模板（15秒）
    GOLDEN_TIMELINE_15S = [
        {"phase": "hook", "start": 0, "end": 3, "description": "钩子画面"},
        {"phase": "content", "start": 3, "end": 10, "description": "快速信息输出"},
        {"phase": "climax", "start": 10, "end": 13, "description": "反转/高潮"},
        {"phase": "cta", "start": 13, "end": 15, "description": "CTA"},
    ]

    def generate(
        self,
        trend: TrendReport | dict[str, Any],
        clips: CandidateClips | dict[str, Any],
        destination: str,
        duration: int = 30,
    ) -> Script:
        """生成旅行 Vlog 剧本

        Args:
            trend: 趋势分析报告
            clips: 候选素材片段
            destination: 目的地
            duration: 目标时长（秒）

        Returns:
            Script: 创意剧本
        """
        logger.info(f"生成剧本: destination={destination}, duration={duration}s")

        # 选择钩子
        hook = self._select_hook(trend, destination, clips)

        # 选择 BGM
        bgm = self._select_bgm(trend, duration)

        # 构建分镜
        timeline = self.GOLDEN_TIMELINE_30S if duration >= 25 else self.GOLDEN_TIMELINE_15S
        scenes = self._build_scenes(timeline, clips, destination, duration)

        # 生成旁白
        voiceover = self._generate_voiceover(scenes, destination, hook)

        # CTA
        cta = self._generate_cta(destination)

        script = Script(
            title=f"{destination}旅行 | {hook.text[:15]}..." if hook.text else f"{destination}旅行",
            destination=destination,
            duration=duration,
            hook=hook,
            bgm_suggestion=bgm,
            voiceover=voiceover,
            scenes=scenes,
            cta=cta,
        )

        logger.info(f"剧本生成完成: {len(scenes)} 个分镜")
        return script

    def generate_short(
        self,
        trend: TrendReport | dict[str, Any],
        clips: CandidateClips | dict[str, Any],
        destination: str,
        duration: int = 15,
        style: str = "fast_cut",
    ) -> Script:
        """生成旅行短视频剧本（15-20秒，高密度快剪）"""
        logger.info(f"生成短视频剧本: destination={destination}, duration={duration}s, style={style}")

        hook = self._select_hook(trend, destination, clips, prefer_shock=True)
        bgm = self._select_bgm(trend, duration, mood="energetic")

        timeline = self.GOLDEN_TIMELINE_15S
        scenes = self._build_scenes(timeline, clips, destination, duration, avg_shot=1.5)

        voiceover = self._generate_voiceover(scenes, destination, hook, short=True)

        script = Script(
            title=f"{destination} | {hook.text[:10]}" if hook.text else f"{destination}",
            destination=destination,
            duration=duration,
            hook=hook,
            bgm_suggestion=bgm,
            voiceover=voiceover,
            scenes=scenes,
            cta="收藏这份攻略",
        )

        return script

    def _select_hook(
        self,
        trend: TrendReport | dict[str, Any],
        destination: str,
        clips: CandidateClips | dict[str, Any],
        prefer_shock: bool = False,
    ) -> Hook:
        """从钩子公式库选择钩子"""
        hook_formulas = []
        if isinstance(trend, TrendReport):
            hook_formulas = trend.hook_formulas
        elif isinstance(trend, dict):
            for hf in trend.get("hook_formulas", []):
                hook_formulas.append(hf)

        if not hook_formulas:
            # 默认钩子
            return Hook(
                type="shock",
                text="",
                duration_sec=3.0,
            )

        # 按效果排序
        sorted_hooks = sorted(
            hook_formulas,
            key=lambda h: h.get("effectiveness", 0) if isinstance(h, dict) else h.effectiveness,
            reverse=True,
        )

        # 如果有航拍/大景素材，优先 shock 型
        if prefer_shock:
            for h in sorted_hooks:
                ht = h.get("hook_type") if isinstance(h, dict) else h.hook_type
                if ht == "shock":
                    return self._build_hook(h, destination)

        # 选择 Top-3 中随机一个（增加多样性）
        selected = sorted_hooks[0]
        return self._build_hook(selected, destination)

    def _build_hook(self, hook_data, destination: str) -> Hook:
        """从 hook formula 数据构建 Hook 对象"""
        if isinstance(hook_data, dict):
            return Hook(
                type=hook_data.get("hook_type", "suspense"),
                text=hook_data.get("template", "").format(
                    destination=destination,
                    N=random.choice([3, 5, 7]),
                    M=random.choice([3, 5, 10]),
                    negative=random.choice(["无聊", "普通", "没什么"]),
                ),
                duration_sec=3.0,
            )
        else:
            return Hook(
                type=hook_data.hook_type,
                text=hook_data.template.format(
                    destination=destination,
                    N=random.choice([3, 5, 7]),
                    M=random.choice([3, 5, 10]),
                    negative=random.choice(["无聊", "普通", "没什么"]),
                ),
                duration_sec=3.0,
            )

    def _select_bgm(self, trend: Any, duration: int, mood: str = "chill") -> BgmSuggestion:
        """选择 BGM"""
        music_trends = []
        if isinstance(trend, TrendReport):
            music_trends = trend.music_trends
        elif isinstance(trend, dict):
            music_trends = trend.get("music_trends", [])

        if music_trends:
            first = music_trends[0]
            if isinstance(first, dict):
                return BgmSuggestion(
                    name=first.get("bgm_name", ""),
                    mood=first.get("mood", mood),
                    bpm=random.randint(80, 120),
                )

        return BgmSuggestion(name="", mood=mood, bpm=random.randint(80, 120))

    def _build_scenes(
        self,
        timeline: list[dict],
        clips: CandidateClips | dict[str, Any],
        destination: str,
        duration: int,
        avg_shot: float = 2.0,
    ) -> list[ScriptScene]:
        """根据时间轴和候选素材构建分镜"""
        scenes = []
        clip_list = []

        if isinstance(clips, CandidateClips):
            clip_list = clips.clips
        elif isinstance(clips, dict):
            for c in clips.get("clips", []):
                clip_list.append(c)

        # 按时间轴分配素材
        clip_idx = 0
        for phase in timeline:
            phase_duration = phase["end"] - phase["start"]
            num_shots = max(1, int(phase_duration / avg_shot))

            for shot in range(num_shots):
                shot_duration = phase_duration / num_shots
                clip = clip_list[clip_idx % len(clip_list)] if clip_list else None

                description = phase["description"]
                if clip:
                    summary = clip.get("scene_summary", "") if isinstance(clip, dict) else clip.scene_summary
                    if summary:
                        description = summary

                narration = self._generate_phase_narration(phase["phase"], destination, shot)

                transition = "cut"
                if phase["phase"] == "intro":
                    transition = "fade"
                elif phase["phase"] == "ending":
                    transition = "fade"
                elif phase["phase"] == "climax" and shot > 0:
                    transition = "dissolve"

                scenes.append(ScriptScene(
                    order=len(scenes) + 1,
                    duration_sec=shot_duration,
                    clip_description=description,
                    narration=narration,
                    subtitle=self._generate_subtitle(phase["phase"], destination, shot),
                    transition=transition,
                    mood=self._phase_mood(phase["phase"]),
                ))
                clip_idx += 1

        return scenes

    def _generate_phase_narration(self, phase: str, destination: str, shot: int) -> str:
        """为每个阶段生成旁白"""
        narrations = {
            "hook": ["", f"{destination}", ""],
            "intro": [f"到了{destination}", f"第一站", f"这里是{destination}"],
            "develop": ["街头的味道", "走走停停", "每一帧都是风景", "烟火气十足"],
            "transit": ["在路上", "下一站", "继续走"],
            "climax": ["值了", "太美了", f"{destination}，再见", "这一刻"],
            "ending": ["", "收藏吧", ""],
            "content": ["这个必去", "推荐", "收藏", "必打卡"],
            "cta": ["", "关注我", ""],
        }

        phase_narrations = narrations.get(phase, [""])
        return phase_narrations[shot % len(phase_narrations)]

    def _generate_subtitle(self, phase: str, destination: str, shot: int) -> str:
        """生成字幕"""
        subtitles = {
            "hook": [destination, "", ""],
            "intro": [destination, "Day 1", ""],
            "develop": ["", "", "", ""],
            "transit": ["", "", ""],
            "climax": [destination, "", ""],
            "ending": ["收藏", "", ""],
            "content": ["必去", "推荐", "收藏", ""],
            "cta": ["收藏这份攻略", "", ""],
        }

        phase_subs = subtitles.get(phase, [""])
        return phase_subs[shot % len(phase_subs)]

    def _phase_mood(self, phase: str) -> str:
        """阶段情绪"""
        moods = {
            "hook": "shock",
            "intro": "curious",
            "develop": "lively",
            "transit": "moving",
            "climax": "epic",
            "ending": "calm",
            "content": "energetic",
            "cta": "engaging",
        }
        return moods.get(phase, "neutral")

    def _generate_voiceover(self, scenes: list[ScriptScene], destination: str, hook: Hook, short: bool = False) -> str:
        """生成旁白全文"""
        parts = []

        if hook.text:
            parts.append(hook.text)

        for scene in scenes:
            if scene.narration:
                parts.append(scene.narration)

        voiceover = "。".join(parts)
        if short:
            # 短视频旁白精简
            words = voiceover[:45]
        else:
            words = voiceover[:90]

        return words

    def _generate_cta(self, destination: str) -> str:
        """生成 CTA"""
        ctas = [
            f"收藏这份{destination}攻略",
            f"你最想去{destination}的哪里？",
            f"关注我，带你看更多{destination}",
            f"评论区告诉我你的{destination}故事",
        ]
        return random.choice(ctas)
