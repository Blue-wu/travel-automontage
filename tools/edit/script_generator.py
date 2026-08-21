"""剧本生成器 — 趋势驱动 + 素材驱动 + 故事化叙事

Agent 在 script_generation 阶段调用此工具，
同时读取 skills/travel-storytelling.md 获取叙事规范。

核心原则：
1. 故事完整性 — 有起承转合，不是素材堆砌
2. 活人感 — 口语化表达，像朋友分享，不是 AI 排比句
3. 利他性 — 有实用信息（攻略/清单/避坑），让观众觉得有价值
"""

from __future__ import annotations

import logging
import random
from typing import Any

from tools.common.models import (
    BgmSuggestion,
    CandidateClips,
    ContentTypeSuggestion,
    Hook,
    SceneCategory,
    Script,
    ScriptScene,
    TrendReport,
)

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """旅行 Vlog 剧本生成器"""

    STORY_BEAT_TEMPLATES = {
        "guide": {
            "beats": [
                {
                    "phase": "hook",
                    "pacing": "fast",
                    "mood": "shock",
                    "shot_type": "extreme_wide",
                    "category_pref": ["sky", "mountain", "lake"],
                    "duration_ratio": 0.1,
                    "narration_style": "pain_point",
                    "subtitle_style": "hook",
                },
                {
                    "phase": "intro",
                    "pacing": "medium",
                    "mood": "curious",
                    "shot_type": "wide",
                    "category_pref": ["road", "architecture"],
                    "duration_ratio": 0.12,
                    "narration_style": "setup",
                    "subtitle_style": "title",
                },
                {
                    "phase": "tip_1",
                    "pacing": "medium",
                    "mood": "helpful",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.18,
                    "narration_style": "tip_numbered",
                    "subtitle_style": "tip",
                },
                {
                    "phase": "tip_2",
                    "pacing": "medium",
                    "mood": "helpful",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.18,
                    "narration_style": "tip_numbered",
                    "subtitle_style": "tip",
                },
                {
                    "phase": "tip_3",
                    "pacing": "medium",
                    "mood": "helpful",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.18,
                    "narration_style": "tip_numbered",
                    "subtitle_style": "tip",
                },
                {
                    "phase": "climax",
                    "pacing": "fast",
                    "mood": "epic",
                    "shot_type": "extreme_wide",
                    "category_pref": ["mountain", "lake", "sunset"],
                    "duration_ratio": 0.12,
                    "narration_style": "payoff",
                    "subtitle_style": "summary",
                },
                {
                    "phase": "cta",
                    "pacing": "slow",
                    "mood": "warm",
                    "shot_type": "wide",
                    "category_pref": ["sunset", "night"],
                    "duration_ratio": 0.12,
                    "narration_style": "call_to_action",
                    "subtitle_style": "cta",
                },
            ],
        },
        "emotional": {
            "beats": [
                {
                    "phase": "hook",
                    "pacing": "slow",
                    "mood": "quiet",
                    "shot_type": "wide",
                    "category_pref": ["sunset", "lake", "sky"],
                    "duration_ratio": 0.12,
                    "narration_style": "feeling",
                    "subtitle_style": "hook",
                },
                {
                    "phase": "arrival",
                    "pacing": "medium",
                    "mood": "curious",
                    "shot_type": "medium",
                    "category_pref": ["road", "architecture"],
                    "duration_ratio": 0.12,
                    "narration_style": "personal",
                    "subtitle_style": "place",
                },
                {
                    "phase": "discovery_1",
                    "pacing": "medium",
                    "mood": "amazed",
                    "shot_type": "wide",
                    "category_pref": [],
                    "duration_ratio": 0.15,
                    "narration_style": "personal_wow",
                    "subtitle_style": "scene",
                },
                {
                    "phase": "discovery_2",
                    "pacing": "medium",
                    "mood": "peaceful",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.15,
                    "narration_style": "personal_feeling",
                    "subtitle_style": "scene",
                },
                {
                    "phase": "discovery_3",
                    "pacing": "medium",
                    "mood": "moved",
                    "shot_type": "wide",
                    "category_pref": [],
                    "duration_ratio": 0.15,
                    "narration_style": "personal_reflection",
                    "subtitle_style": "scene",
                },
                {
                    "phase": "climax",
                    "pacing": "slow",
                    "mood": "emotional",
                    "shot_type": "extreme_wide",
                    "category_pref": ["sunset", "mountain", "night"],
                    "duration_ratio": 0.16,
                    "narration_style": "insight",
                    "subtitle_style": "insight",
                },
                {
                    "phase": "ending",
                    "pacing": "slow",
                    "mood": "calm",
                    "shot_type": "wide",
                    "category_pref": ["night", "sunset"],
                    "duration_ratio": 0.15,
                    "narration_style": "closing",
                    "subtitle_style": "cta",
                },
            ],
        },
        "informational": {
            "beats": [
                {
                    "phase": "hook",
                    "pacing": "fast",
                    "mood": "curious",
                    "shot_type": "medium",
                    "category_pref": ["architecture", "road"],
                    "duration_ratio": 0.08,
                    "narration_style": "question",
                    "subtitle_style": "hook",
                },
                {
                    "phase": "intro",
                    "pacing": "fast",
                    "mood": "informative",
                    "shot_type": "wide",
                    "category_pref": ["sky", "mountain"],
                    "duration_ratio": 0.1,
                    "narration_style": "list_intro",
                    "subtitle_style": "title",
                },
                {
                    "phase": "point_1",
                    "pacing": "medium",
                    "mood": "informative",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.16,
                    "narration_style": "fact_numbered",
                    "subtitle_style": "number",
                },
                {
                    "phase": "point_2",
                    "pacing": "medium",
                    "mood": "informative",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.16,
                    "narration_style": "fact_numbered",
                    "subtitle_style": "number",
                },
                {
                    "phase": "point_3",
                    "pacing": "medium",
                    "mood": "surprised",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.16,
                    "narration_style": "fact_surprise",
                    "subtitle_style": "number",
                },
                {
                    "phase": "point_4",
                    "pacing": "medium",
                    "mood": "informative",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.14,
                    "narration_style": "fact_numbered",
                    "subtitle_style": "number",
                },
                {
                    "phase": "ending",
                    "pacing": "fast",
                    "mood": "helpful",
                    "shot_type": "wide",
                    "category_pref": ["sunset", "night"],
                    "duration_ratio": 0.1,
                    "narration_style": "save_for_later",
                    "subtitle_style": "cta",
                },
            ],
        },
        "vlog": {
            "beats": [
                {
                    "phase": "hook",
                    "pacing": "medium",
                    "mood": "casual",
                    "shot_type": "medium",
                    "category_pref": ["road", "architecture"],
                    "duration_ratio": 0.1,
                    "narration_style": "hey_guys",
                    "subtitle_style": "hook",
                },
                {
                    "phase": "morning",
                    "pacing": "medium",
                    "mood": "fresh",
                    "shot_type": "medium",
                    "category_pref": ["sky", "architecture"],
                    "duration_ratio": 0.12,
                    "narration_style": "day_start",
                    "subtitle_style": "time",
                },
                {
                    "phase": "activity_1",
                    "pacing": "medium",
                    "mood": "happy",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.15,
                    "narration_style": "activity_story",
                    "subtitle_style": "place",
                },
                {
                    "phase": "activity_2",
                    "pacing": "medium",
                    "mood": "excited",
                    "shot_type": "medium",
                    "category_pref": [],
                    "duration_ratio": 0.15,
                    "narration_style": "activity_story",
                    "subtitle_style": "place",
                },
                {
                    "phase": "food",
                    "pacing": "medium",
                    "mood": "happy",
                    "shot_type": "close_up",
                    "category_pref": ["food"],
                    "duration_ratio": 0.12,
                    "narration_style": "food_reaction",
                    "subtitle_style": "food",
                },
                {
                    "phase": "sunset_view",
                    "pacing": "slow",
                    "mood": "peaceful",
                    "shot_type": "wide",
                    "category_pref": ["sunset", "mountain", "lake"],
                    "duration_ratio": 0.16,
                    "narration_style": "day_end_thought",
                    "subtitle_style": "scene",
                },
                {
                    "phase": "night_ending",
                    "pacing": "slow",
                    "mood": "warm",
                    "shot_type": "medium",
                    "category_pref": ["night", "architecture"],
                    "duration_ratio": 0.1,
                    "narration_style": "day_end_summary",
                    "subtitle_style": "cta",
                },
            ],
        },
    }

    NARRATION_TEMPLATES = {
        "pain_point": [
            "去了{destination}才发现，之前的攻略全是坑",
            "很多人去{destination}都踩了这个坑，包括我",
            "如果你打算去{destination}，这个视频一定要看完",
        ],
        "setup": [
            "这次我花了3天时间，帮你们把{destination}的路线踩明白了",
            "这是我第3次来{destination}，总结了3条最实用的建议",
            "刚来{destination}的时候，我也完全不知道从哪玩起",
        ],
        "tip_numbered": [
            "第{num}个，{content}",
            "第{num}点要注意的是，{content}",
            "还有第{num}个，{content}",
        ],
        "payoff": [
            "所以你看，{destination}真的值得来一次",
            "当这些风景出现在眼前的时候，你会觉得一切都值了",
            "这就是为什么那么多人来过{destination}就不想走了",
        ],
        "call_to_action": [
            "收藏起来，去{destination}的时候用得上",
            "还有什么问题，评论区问我",
            "觉得有用的话，点个赞再走",
        ],
        "feeling": [
            "说实话，来{destination}之前，我没想到会是这样",
            "有些地方，真的要自己去了才懂",
            "站在这里的那一刻，突然就不想说话了",
        ],
        "personal": [
            "今天是我在{destination}的第一天",
            "坐了8个小时的车，终于到了",
            "从出发到现在，一路都在惊喜",
        ],
        "personal_wow": [
            "哇，这也太好看了吧",
            "说实话，我被震撼到了",
            "亲眼看到的那一刻，真的有点不敢相信",
        ],
        "personal_feeling": [
            "站在这里，感觉整个人都静下来了",
            "有时候真的需要这样的时刻",
            "风一吹，什么烦恼都没了",
        ],
        "personal_reflection": [
            "以前总想着要去很多地方",
            "现在觉得，能安安静静待一会儿就挺好",
            "旅行的意义，大概就是找到自己吧",
        ],
        "insight": [
            "原来最好的风景，真的都在路上",
            "有些东西，照片真的拍不出来",
            "那一刻我突然明白了，为什么那么多人热爱旅行",
        ],
        "closing": [
            "如果你也累了，就出来走走吧",
            "{destination}，我一定还会再来的",
            "这趟旅程，值了",
        ],
        "question": [
            "你知道{destination}有多少好玩的地方吗？",
            "关于{destination}，这几件事你听说过吗？",
            "90%的人都不知道的{destination}冷知识",
        ],
        "list_intro": [
            "今天跟大家盘点一下，{destination}最值得去的几个地方",
            "关于{destination}的几件事，最后一个最让人意外",
            "一分钟，带你了解{destination}",
        ],
        "fact_numbered": [
            "第{num}个，{content}",
            "第{num}点，{content}",
            "还有第{num}个，{content}",
        ],
        "fact_surprise": [
            "最让人意外的是第{num}个，{content}",
            "你肯定想不到，第{num}个居然是这样的",
            "说到第{num}个，很多人都不知道",
        ],
        "save_for_later": [
            "收藏起来，以后去{destination}用得上",
            "你还知道哪些？评论区补充",
            "关注我，带你看更多好地方",
        ],
        "hey_guys": [
            "哈喽，我现在在{destination}",
            "今天带大家云游{destination}",
            "辞职旅行第N天，我来到了{destination}",
        ],
        "day_start": [
            "今天起了个大早，准备出去玩",
            "新的一天，从{destination}的早晨开始",
            "今天的行程很满，跟着我一起吧",
        ],
        "activity_story": [
            "然后我们就来了{content}",
            "下一站，{content}",
            "没想到这里还有{content}，太惊喜了",
        ],
        "food_reaction": [
            "这个也太好吃了吧",
            "来{destination}一定要吃这个",
            "为了这口吃的，跑这么远都值",
        ],
        "day_end_thought": [
            "看着夕阳，突然觉得好治愈",
            "今天好累，但也好开心",
            "这样的日子，真希望能多几天",
        ],
        "day_end_summary": [
            "今天就到这里啦，明天继续",
            "晚安，{destination}",
            "今天真的是超棒的一天",
        ],
    }

    SUBTITLE_TEMPLATES = {
        "hook": ["{destination}", "这里是{destination}"],
        "title": ["{destination}攻略", "旅行清单"],
        "tip": ["实用建议", "避坑指南", "小Tips"],
        "summary": ["{destination}值得"],
        "cta": ["收藏备用", "点赞收藏"],
        "place": ["{name}", "打卡{name}"],
        "scene": ["治愈时刻", "风景如画", "绝美"],
        "insight": ["旅行的意义", "人间值得"],
        "time": ["Day 1", "出发啦", "早安"],
        "food": ["当地美食", "太好吃了"],
        "number": ["NO.{num}", "第{num}个"],
    }

    def generate(
        self,
        trend: TrendReport | dict[str, Any],
        clips: CandidateClips | dict[str, Any],
        destination: str,
        duration: int = 0,
    ) -> Script:
        """生成旅行 Vlog 剧本（素材驱动 + 趋势驱动）

        Args:
            trend: 趋势分析报告
            clips: 候选素材（带分类）
            destination: 目的地
            duration: 目标时长，0 表示素材驱动自动决定

        Returns:
            Script: 创意剧本
        """
        logger.info(f"生成剧本: destination={destination}, duration={duration}s (0=auto)")

        # 强制使用情感共鸣型剧本（更有共鸣感、更走心）
        from tools.common.models import ContentTypeSuggestion
        content_type = ContentTypeSuggestion(
            type="emotional",
            name="情感型",
            description="情感共鸣型，用走心文案打动观众",
            popularity=0.9,
            storytelling_tips=["用画面+音乐营造氛围", "第一人称视角", "情绪曲线"],
            hooks=[
                f"去了{destination}才知道，什么叫人间值得",
                f"有些地方，真的要自己去了才懂",
                f"站在{destination}的那一刻，突然就治愈了",
                f"你有没有过，看到风景就想哭的瞬间",
                f"总要有一次，为了{destination}奔赴千里",
                f"如果累了，就去{destination}走走吧",
            ],
        )

        categories = self._extract_categories(clips)
        available_clips = self._extract_clips(clips)
        self._clips_cache = clips

        meaningful_categories = [c for c in categories if c.category != "other"]
        if len(meaningful_categories) < 2:
            logger.info("素材分类不足，自动切换为情感型内容")
            from tools.common.models import ContentTypeSuggestion
            content_type = ContentTypeSuggestion(
                type="emotional",
                name="情感型",
                description="素材不足时默认情感型，更通用自然",
                popularity=0.75,
                storytelling_tips=["用画面+音乐营造氛围", "第一人称视角", "情绪曲线"],
                hooks=[
                    f"去了{destination}才知道，什么叫人间值得",
                    f"有些地方，真的要自己去了才懂",
                    f"站在{destination}的那一刻，突然就治愈了",
                ],
            )

        if duration <= 0:
            duration = self._calc_optimal_duration(clips)
            logger.info(f"素材驱动自动决定时长: {duration:.0f}s")

        beat_template = self.STORY_BEAT_TEMPLATES.get(
            "emotional",
            self.STORY_BEAT_TEMPLATES["emotional"],
        )

        hook = self._build_hook(content_type, destination)
        bgm = self._build_bgm(content_type)
        scenes = self._build_scenes(
            beat_template["beats"],
            categories,
            content_type,
            destination,
            duration,
        )

        voiceover = self._generate_voiceover(scenes, hook)
        cta = self._build_cta(content_type, destination)

        title = self._build_title(content_type, destination)

        script = Script(
            title=title,
            destination=destination,
            duration=round(duration, 1),
            hook=hook,
            bgm_suggestion=bgm,
            voiceover=voiceover,
            scenes=scenes,
            cta=cta,
        )

        logger.info(f"剧本生成完成: {len(scenes)} 个分镜, 内容类型={content_type.name}")
        return script

    def _select_content_type(self, trend: TrendReport | dict) -> ContentTypeSuggestion:
        """根据趋势分析选择内容类型"""
        if isinstance(trend, TrendReport) and trend.recommended_content_type:
            return trend.recommended_content_type

        if isinstance(trend, dict):
            recommended = trend.get("recommended_content_type")
            if recommended:
                return ContentTypeSuggestion(**recommended)

        from tools.common.models import ContentTypeSuggestion
        return ContentTypeSuggestion(
            type="emotional",
            name="情感型",
            description="默认情感型，容易引起共鸣",
            popularity=0.7,
        )

    def _extract_categories(self, clips: CandidateClips | dict) -> list[SceneCategory]:
        """提取素材分类"""
        if isinstance(clips, CandidateClips):
            return clips.categories
        if isinstance(clips, dict):
            cats = clips.get("categories", [])
            from tools.common.models import SceneCategory
            return [SceneCategory(**c) for c in cats]
        return []

    def _extract_clips(self, clips: CandidateClips | dict) -> list:
        """提取素材列表"""
        if isinstance(clips, CandidateClips):
            return clips.clips
        if isinstance(clips, dict):
            return clips.get("clips", [])
        return []

    def _calc_optimal_duration(self, clips: CandidateClips | dict) -> float:
        """计算最佳输出时长"""
        if isinstance(clips, CandidateClips):
            estimated = clips.estimated_output_duration
            if estimated > 0:
                return estimated

        if isinstance(clips, dict):
            estimated = clips.get("estimated_output_duration", 0)
            if estimated > 0:
                return estimated

        return 30.0

    def _build_hook(self, content_type: ContentTypeSuggestion, destination: str) -> Hook:
        """构建钩子"""
        if content_type.hooks:
            hook_text = random.choice(content_type.hooks)
        elif content_type.type == "emotional":
            hook_options = [
                f"去了{destination}才知道，什么叫人间值得",
                f"有些地方，真的要自己去了才懂",
                f"站在{destination}的那一刻，突然就治愈了",
                f"你有没有过，看到风景就想哭的瞬间",
                f"总要有一次，为了{destination}奔赴千里",
                f"如果累了，就去{destination}走走吧",
            ]
            hook_text = random.choice(hook_options)
        else:
            hook_text = f"{destination}也太美了吧"

        hook_type = content_type.type
        if hook_type == "guide":
            hook_type = "suspense"
        elif hook_type == "informational":
            hook_type = "question"
        elif hook_type == "vlog":
            hook_type = "resonance"

        return Hook(
            type=hook_type,
            text=hook_text,
            duration_sec=2.5,
        )

    def _build_bgm(self, content_type: ContentTypeSuggestion) -> BgmSuggestion:
        """构建 BGM 建议"""
        bgm_map = {
            "guide": {"name": "轻快旅行BGM", "mood": "cheerful", "bpm": 110},
            "emotional": {"name": "治愈旅行BGM", "mood": "chill", "bpm": 85},
            "informational": {"name": "活力BGM", "mood": "energetic", "bpm": 120},
            "vlog": {"name": "日常Vlog BGM", "mood": "casual", "bpm": 100},
        }

        info = bgm_map.get(content_type.type, bgm_map["emotional"])
        return BgmSuggestion(**info)

    def _build_title(self, content_type: ContentTypeSuggestion, destination: str) -> str:
        """构建标题"""
        title_templates = {
            "guide": [f"{destination}避坑指南，看完再去", f"去{destination}之前一定要看", f"{destination}实用攻略"],
            "emotional": [f"去了{destination}才懂什么叫治愈", f"{destination}，来了就不想走", f"这就是{destination}的魅力"],
            "informational": [f"关于{destination}的几件事", f"{destination}最值得去的地方", f"一分钟了解{destination}"],
            "vlog": [f"我的{destination}之旅", f"一个人的{destination}", f"在{destination}的一天"],
        }

        templates = title_templates.get(content_type.type, title_templates["emotional"])
        return random.choice(templates)

    def _build_cta(self, content_type: ContentTypeSuggestion, destination: str) -> str:
        """构建 CTA 结尾号召"""
        cta_templates = {
            "guide": [
                f"收藏起来，去{destination}的时候用得上",
                f"还有什么问题，评论区问我",
                f"觉得有用的话，点个赞再走",
            ],
            "emotional": [
                f"如果你也累了，就去{destination}走走吧",
                f"{destination}，你一定要来一次",
                "评论区告诉我，你最想去哪里",
            ],
            "informational": [
                f"收藏起来，以后去{destination}用得上",
                "你还知道哪些？评论区补充",
                "关注我，带你看更多好地方",
            ],
            "vlog": [
                "明天继续更新，记得来看",
                f"你们觉得{destination}怎么样？评论区告诉我",
                "喜欢的话点个关注吧",
            ],
        }

        templates = cta_templates.get(content_type.type, cta_templates["emotional"])
        return random.choice(templates)

    def _build_scenes(
        self,
        beats: list[dict],
        categories: list[SceneCategory],
        content_type: ContentTypeSuggestion,
        destination: str,
        total_duration: float,
    ) -> list[ScriptScene]:
        """根据故事节拍构建分镜

        每个分镜会从对应类别中选一个具体素材，
        用素材的真实标签和摘要生成精准文案。
        """
        scenes = []
        available_categories = {c.category: c for c in categories}
        all_clips = self._extract_clips(self._clips_cache) if hasattr(self, "_clips_cache") else []

        tip_counter = 1
        point_counter = 1
        used_clips: set[str] = set()
        category_usage: dict[str, int] = {}
        generic_categories = {"people", "aerial"}

        for i, beat in enumerate(beats):
            duration = total_duration * beat["duration_ratio"]

            preferred_cats = beat.get("category_pref", [])
            matched_cat = self._pick_category(
                preferred_cats, available_categories, category_usage, generic_categories
            )

            if matched_cat:
                category_usage[matched_cat.category] = category_usage.get(matched_cat.category, 0) + 1

            selected_clip = self._pick_clip_for_scene(matched_cat, used_clips)
            if selected_clip:
                clip_key = f"{selected_clip.asset_id}_{selected_clip.start_sec}"
                used_clips.add(clip_key)

            visual_keywords = self._get_visual_keywords(matched_cat, selected_clip)

            narration = self._generate_narration(
                beat["narration_style"],
                destination,
                matched_cat,
                tip_counter if content_type.type == "guide" else point_counter,
                content_type.type,
                selected_clip,
            )
            subtitle = self._generate_subtitle(
                beat["subtitle_style"],
                destination,
                matched_cat,
                tip_counter if content_type.type in ("guide", "informational") else None,
                selected_clip,
            )

            if beat["narration_style"] in ("tip_numbered", "fact_numbered", "fact_surprise"):
                if content_type.type == "guide":
                    tip_counter += 1
                else:
                    point_counter += 1

            scenes.append(ScriptScene(
                order=len(scenes) + 1,
                duration_sec=round(duration, 2),
                clip_description=beat["phase"],
                visual_keywords=visual_keywords,
                narration=narration,
                subtitle=subtitle,
                transition=self._pick_transition(i, beat["pacing"]),
                mood=beat["mood"],
                shot_type=beat["shot_type"],
            ))

        return scenes

    def _pick_clip_for_scene(
        self, category: SceneCategory | None, used_clips: set[str]
    ):
        """为分镜挑选一个合适的素材片段（尽量不重复）"""
        if category is None or not category.clips:
            return None

        clips = category.clips
        unused = [c for c in clips if f"{c.asset_id}_{c.start_sec}" not in used_clips]

        pool = unused if unused else clips
        pool = sorted(pool, key=lambda c: -c.quality)

        top_n = min(5, len(pool))
        if top_n == 0:
            return None

        return random.choice(pool[:top_n])

    def _pick_category(
        self,
        preferred_cats: list[str],
        available: dict[str, SceneCategory],
        category_usage: dict[str, int] | None = None,
        generic_categories: set[str] | None = None,
    ) -> SceneCategory | None:
        """根据偏好选择可用的素材类别

        优先级：
        1. 偏好列表中存在、且有素材的类别（优先选没用过的）
        2. 所有可用类别中，使用次数最少、且非通用类别的
        3. 兜底：使用次数最少的
        """
        if category_usage is None:
            category_usage = {}
        if generic_categories is None:
            generic_categories = set()

        # 1. 先在偏好列表里找
        best_preferred = None
        best_preferred_usage = 999
        for cat_key in preferred_cats:
            if cat_key in available and available[cat_key].count > 0:
                usage = category_usage.get(cat_key, 0)
                if usage < best_preferred_usage:
                    best_preferred_usage = usage
                    best_preferred = available[cat_key]
        if best_preferred and best_preferred_usage == 0:
            return best_preferred

        # 2. 找所有非通用类别中使用最少的
        best_scenic = None
        best_scenic_usage = 999
        for cat in available.values():
            if cat.count == 0:
                continue
            if cat.category in generic_categories:
                continue
            usage = category_usage.get(cat.category, 0)
            if usage < best_scenic_usage:
                best_scenic_usage = usage
                best_scenic = cat
        if best_scenic:
            return best_scenic

        # 3. 兜底：从所有类别（包括通用）里找使用最少的
        best_any = None
        best_any_usage = 999
        for cat in available.values():
            if cat.count == 0:
                continue
            usage = category_usage.get(cat.category, 0)
            if usage < best_any_usage:
                best_any_usage = usage
                best_any = cat
        if best_any:
            return best_any

        return None

    def _get_visual_keywords(self, category: SceneCategory | None, clip=None) -> list[str]:
        """获取视觉关键词 —— 只用素材里真实存在的物件

        原实现内嵌了一份 类别→关键词 查表（和 clip_classifier.VISUAL_TAG_CANDIDATES
        是两份互不同步的表），命中时返回的是类别同义词而非画面内容，
        对文案毫无价值："湖泊/湛蓝/倒影" 写不出任何有意思的东西。
        现在改为：拿得到具体物件就用，拿不到就返回空，让下游走留白而不是套话。
        """
        if clip is None:
            return []
        # subjects 优先 —— 那才是具体物件（skills/travel-copywriting.md 手法 A 抓手）
        subjects = list(getattr(clip, "subjects", None) or [])
        if subjects:
            return subjects[:6]
        tags = list(getattr(clip, "visual_tags", None) or [])
        return tags[:6]

    def _generate_narration(
        self,
        style: str,
        destination: str,
        category: SceneCategory | None,
        num: int,
        content_type: str,
        clip=None,
    ) -> str:
        """生成旁白文案（活人感 + 画面精准对应）

        优先使用真实素材的标签和摘要生成文案，
        没有时回退到模板生成。
        """
        if clip and (clip.visual_tags or clip.scene_summary) and clip.scene_summary and clip.scene_summary != "片段 1":
            return self._generate_narration_from_clip(style, destination, clip, num)

        templates = self.NARRATION_TEMPLATES.get(style, [""])
        template = random.choice(templates) if templates else ""

        if category and category.category != "other":
            content_desc = category.display_name
        else:
            content_desc = random.choice([
                "这样的风景",
                "眼前这一切",
                "这里的美",
                "这样的画面",
                "这般景色",
            ])

        try:
            text = template.format(destination=destination, num=num, content=content_desc)
        except (KeyError, IndexError):
            text = template

        return text

    def _generate_narration_from_clip(self, style: str, destination: str, clip, num: int) -> str:
        """基于真实素材内容生成旁白 — 口语化、有活人感、不AI"""
        tags = clip.visual_tags or []
        summary = clip.scene_summary or ""

        generic_tags = {"人物", "人像", "航拍", "俯视", "当地"}
        scenic_tags = [t for t in tags if t not in generic_tags]
        tag0 = scenic_tags[0] if scenic_tags else (tags[0] if tags else "风景")
        tag1 = scenic_tags[1] if len(scenic_tags) > 1 else (tags[1] if len(tags) > 1 else "")
        tag_str = "、".join([t for t in tags[:3] if t not in generic_tags][:2]) if tags else "风景"
        if not tag_str:
            tag_str = tag0

        # ── 钩子 / 开场类 ──────────────────────────────────────
        if style == "pain_point":
            options = [
                f"去了{destination}才知道，之前做的攻略全是坑",
                f"如果你打算去{destination}，这几个坑千万别踩",
                f"{destination}最容易踩的坑，我帮你踩过了",
            ]
            return random.choice(options)

        if style == "hey_guys":
            options = [
                f"今天带你们云游{destination}，全程高能",
                f"{destination}之旅开始了，跟我走吧",
                f"出发去{destination}啦，看看一路上能遇到什么",
            ]
            return random.choice(options)

        if style == "question":
            options = [
                f"你知道{destination}最值得去的地方是哪吗？",
                f"去过{destination}的人，都会推荐这个地方",
                f"如果只能去一个地方，{destination}你选哪？",
            ]
            return random.choice(options)

        if style == "setup":
            options = [
                f"这次在{destination}待了一周，总结了最干货的经验",
                f"{destination}我去了三次，每次都有新发现",
                f"为了拍这集，我在{destination}跑了上千公里",
            ]
            return random.choice(options)

        if style == "arrival":
            options = [
                f"终于到{destination}了，比想象中还美",
                f"坐了好久的车，看到这{tag0}的瞬间，值了",
                f"{destination}，我终于来了",
                f"刚到{destination}就被这{tag0}震撼到了",
            ]
            return random.choice(options)

        if style == "day_start":
            options = [
                f"今天的{destination}天气超好，出发！",
                f"一早起来就看到这么美的{tag0}，今天值了",
                f"在{destination}的第一天，从这片{tag0}开始",
            ]
            return random.choice(options)

        if style == "list_intro":
            options = [
                f"去{destination}必看的几个地方，最后一个最绝",
                f"{destination}这几个地方，错过一个都可惜",
                f"我心目中{destination}的TOP{num or 3}，你去过几个？",
            ]
            return random.choice(options)

        # ── 干货 / 攻略类 ──────────────────────────────────────
        if style in ("tip_numbered", "fact_numbered"):
            options = [
                f"第{num}个必去的就是{tag0}，真的超出片",
                f"第{num}站，{tag0}，来{destination}千万别错过",
                f"还有第{num}个，就是这片{tag_str}，超级震撼",
                f"第{num}个推荐：{tag0}，去过的人都说值",
            ]
            return random.choice(options)

        if style == "fact_surprise":
            options = [
                f"你敢信吗？{tag0}居然在{destination}也能看到",
                f"最让我意外的是这片{tag0}，完全没想到",
                f"说出来你可能不信，{destination}还有这样的{tag0}",
            ]
            return random.choice(options)

        if style == "save_for_later":
            options = [
                f"记得点赞收藏，去{destination}的时候用得上",
                f"这篇攻略先存着，下次去{destination}直接抄作业",
                f"收藏起来，{destination}旅行绝对用得到",
            ]
            return random.choice(options)

        # ── 感受 / 情绪类 ──────────────────────────────────────
        if style == "feeling":
            options = [
                f"风吹过的时候，看着{tag_str}，突然觉得什么都不重要了",
                f"站在这{tag0}面前，心里特别安静",
                f"那一刻，世界只剩下风和眼前的{tag0}",
                f"你有没有过那种，看到风景就想哭的瞬间",
                f"有些风景，看一眼就记一辈子",
            ]
            return random.choice(options)

        if style == "personal":
            options = [
                f"说实话，来{destination}之前我没抱太大期望",
                f"这是我第一次来{destination}，比想象中震撼太多",
                f"出发前做了很多攻略，但真站在这里还是懵了",
                f"之前总在视频里看{destination}，今天终于亲眼见到了",
                f"我以为我不会惊讶，但看到这{tag0}还是呆住了",
            ]
            return random.choice(options)

        if style == "wow":
            options = [
                f"我的天，这{tag0}也太绝了吧",
                f"你看这{tag_str}，是不是跟画一样",
                f"这就是真实存在的{tag0}吗？太不真实了",
                f"我天，这{tag0}一眼望不到头",
            ]
            return random.choice(options)

        if style == "personal_wow":
            options = [
                f"我跟你说，亲眼看到这{tag0}的时候，整个人都麻了",
                f"说实话，这{tag0}比任何视频里都震撼",
                f"我当时站在那，看着{tag_str}，半天说不出一句话",
                f"那一刻我明白了，为什么那么多人一定要来{destination}",
                f"相机拍不出它万分之一的美，真的",
            ]
            return random.choice(options)

        if style == "personal_feeling":
            options = [
                f"我在这坐了好久，什么都没做，就看着{tag0}",
                f"说实话，这是我这半年来最放松的一刻",
                f"每次觉得累的时候，就会想起这片{tag0}",
                f"原来真的有地方，能让人瞬间平静下来",
                f"你有没有试过，对着一片{tag0}发呆一整个下午",
            ]
            return random.choice(options)

        if style == "calm_reflection":
            options = [
                f"看着{tag_str}，突然觉得生活里那些事都不算什么了",
                f"人在大自然面前，真的太渺小了",
                f"有些路，要自己走过才知道是什么感觉",
                f"旅行不是为了赶路，是为了停下来看看{tag0}",
                f"时间在这好像变慢了，挺好的",
            ]
            return random.choice(options)

        if style == "personal_reflection":
            options = [
                f"走了这么远，突然想明白了很多事",
                f"以前总想着要快点到达，现在觉得路上才是最好的",
                f"看着这片{tag0}，突然就和自己和解了",
                f"有时候觉得，人生就该多看看这样的{tag0}",
                f"出来走走才发现，世界比想象中大得多",
            ]
            return random.choice(options)

        if style == "resonance":
            options = [
                f"旅行的意义，大概就是这样的时刻吧",
                f"有些风景，真的要自己来看过才懂",
                f"那一刻突然觉得，再远都值得",
                f"你说，人为什么总想去看看远方呢",
            ]
            return random.choice(options)

        if style == "activity_story":
            if tags:
                options = [
                    f"今天开了好久的车，就是为了看这{tag0}",
                    f"为了拍这片{tag0}，我在这等了两个小时",
                    f"终于见到了传说中的{tag0}，名不虚传",
                    f"一路颠簸，但看到{tag_str}的瞬间，值了",
                ]
                return random.choice(options)
            return "这一路的风景，真的值得"

        if style == "food_reaction":
            options = [
                f"来了{destination}一定要尝尝当地的美食，太香了",
                f"这就是{destination}的特色美食，绝了",
                f"为了这口吃的，跑这么远都值",
            ]
            return random.choice(options)

        # ── 高潮 / 结尾类 ──────────────────────────────────────
        if style == "payoff":
            options = [
                f"走了这么远，就是为了这一刻的{tag0}",
                f"终于看到了心心念念的{tag_str}，眼泪都快出来了",
                f"看到这片{tag0}的瞬间，觉得所有的奔波都值了",
                f"这就是我梦里的{destination}，一模一样",
            ]
            return random.choice(options)

        if style == "climax":
            options = [
                f"你看，这就是{destination}的{tag0}",
                f"我到现在还记得第一次看到{tag0}的那种感觉",
                f"眼前的{tag_str}，大到让人失语",
                f"来过这里，才算真的到过{destination}",
            ]
            return random.choice(options)

        if style == "insight":
            options = [
                f"其实{destination}的美，不在攻略里，在路上",
                f"旅行最棒的从来不是终点，是路上的{tag0}",
                f"人这一辈子，一定要去看看{destination}的{tag0}",
                f"有些地方去了会后悔一阵子，但{destination}不去会后悔一辈子",
                f"你问我旅行的意义是什么，大概就是遇见这样的{tag0}吧",
            ]
            return random.choice(options)

        if style == "day_end_thought":
            options = [
                f"今天快要结束了，但这片{tag0}会记很久",
                f"临睡前还在想今天看到的{tag0}，有点不真实",
                f"又是被{destination}治愈的一天",
                f"如果可以，真想每天都能看到这样的{tag0}",
            ]
            return random.choice(options)

        if style == "day_end_summary":
            options = [
                f"今天的{destination}，从{tag0}开始，到{tag1 or '星星'}结束",
                f"这一天，值了",
                f"在{destination}的第一天，比想象中好太多",
            ]
            return random.choice(options)

        if style == "closing":
            options = [
                f"{destination}，我们还会再见的",
                f"故事到这里，但{destination}的美远不止这些",
                f"这趟旅程结束了，但有些东西会一直带着",
                f"如果你也喜欢这样的{destination}，我们路上见",
            ]
            return random.choice(options)

        if style == "call_to_action":
            options = [
                f"如果你也想去{destination}，有问题评论区问我",
                f"觉得不错的话，点个赞吧，下次带你看更多{destination}的风景",
                f"关注我，一起去看更多像{tag0}这样的风景",
            ]
            return random.choice(options)

        if style == "cta_relax":
            options = [
                f"如果你也累了，就来{destination}看看{tag0}吧",
                f"压力大的时候，就找个时间去看看这样的风景",
                f"别总忙着赶路，偶尔也停下来看看{tag0}",
                f"找个时间出发吧，{destination}等你",
            ]
            return random.choice(options)

        # 兜底：用 tag 拼一句自然的话
        if tags:
            options = [
                f"眼前的{tag_str}，真的太美了",
                f"这就是{destination}的{tag0}，名不虚传",
                f"看到这片{tag0}，什么都值了",
            ]
            return random.choice(options)

        return "这里真的太美了"

    def _generate_subtitle(
        self,
        style: str,
        destination: str,
        category: SceneCategory | None,
        num: int | None,
        clip=None,
    ) -> str:
        """生成字幕（优先用真实素材内容）"""
        if clip and (clip.visual_tags or clip.scene_summary):
            result = self._generate_subtitle_from_clip(style, destination, clip, num)
            if result:
                return result

        templates = self.SUBTITLE_TEMPLATES.get(style, [""])
        template = random.choice(templates) if templates else ""

        if category and category.category != "other":
            name = category.display_name
        else:
            name = destination

        try:
            if num is not None:
                text = template.format(destination=destination, name=name, num=num)
            else:
                text = template.format(destination=destination, name=name)
        except (KeyError, IndexError):
            text = template

        return text

    def _generate_subtitle_from_clip(self, style: str, destination: str, clip, num: int | None) -> str:
        """基于真实素材内容生成字幕 — 短、有共鸣、金句式"""
        tags = clip.visual_tags or []

        generic_tags = {"人物", "人像", "航拍", "俯视", "当地"}
        scenic_tags = [t for t in tags if t not in generic_tags]
        tag0 = scenic_tags[0] if scenic_tags else (tags[0] if tags else "风景")
        tag1 = scenic_tags[1] if len(scenic_tags) > 1 else (tags[1] if len(tags) > 1 else "")

        # ── 钩子类 ──────────────────────────────────────
        if style in ("hook", "feeling"):
            options = [
                "人间值得",
                "治愈瞬间",
                "这就是远方",
                f"{destination}的样子",
                "来了就懂了",
            ]
            return random.choice(options)

        # ── 个人感受类 ──────────────────────────────────────
        if style == "personal":
            options = [
                f"初见{destination}",
                "比想象中震撼",
                "终于到了",
                "亲眼所见",
            ]
            return random.choice(options)

        if style in ("wow", "personal_wow"):
            options = [
                f"{tag0}·太绝了",
                "整个人都麻了",
                "亲眼见到才懂",
                f"{tag0}名不虚传",
            ]
            return random.choice(options)

        if style in ("place", "scene"):
            if tag0:
                options = [
                    f"{tag0}的{destination}",
                    f"遇见{tag0}",
                    f"{tag0}的风",
                    f"这里是{tag0}",
                ]
                return random.choice(options)
            return destination

        # ── 情感共鸣类 ──────────────────────────────────────
        if style in ("personal_feeling", "calm_reflection"):
            options = [
                "什么都不想了",
                "时间慢下来了",
                "被治愈的一天",
                "安静且自由",
                "风知道答案",
            ]
            return random.choice(options)

        if style in ("personal_reflection", "insight"):
            options = [
                "路上才是答案",
                "人在自然面前",
                "走了很远的路",
                "和自己和解了",
                "世界比想象中大",
            ]
            return random.choice(options)

        # ── 高潮类 ──────────────────────────────────────
        if style in ("climax", "payoff", "emotional_peak"):
            options = [
                f"{destination}的王炸",
                "所有奔波都值了",
                "梦里的样子",
                f"{tag0}·一眼万年",
                "来过就不后悔",
            ]
            return random.choice(options)

        # ── 结尾类 ──────────────────────────────────────
        if style in ("closing", "day_end_thought", "day_end_summary"):
            options = [
                "还会再来的",
                "故事还很长",
                "带着风景离开",
                f"再见{destination}",
            ]
            return random.choice(options)

        if style in ("cta", "call_to_action", "cta_relax"):
            options = [
                "你也去吧",
                "路上见",
                "出发永远不晚",
                "点赞收藏",
            ]
            return random.choice(options)

        if style == "title":
            return f"{destination}·治愈之旅"

        if style == "time":
            return "新的一天"

        # ── 攻略类兜底（保留但优化） ───────────────────────────
        if style in ("tip", "number", "tip_numbered", "fact_numbered"):
            if tag0 and num:
                return f"第{num}站·{tag0}"
            if tag0:
                return tag0
            return f"第{num}个" if num else ""

        if tag0:
            return tag0

        return ""

    def _pick_transition(self, index: int, pacing: str) -> str:
        """选择转场效果"""
        if pacing == "fast":
            return "cut"
        elif pacing == "slow" and index > 0:
            return "fade"
        else:
            return "dissolve" if index % 3 == 0 else "cut"

    def _generate_voiceover(self, scenes: list[ScriptScene], hook: Hook) -> str:
        """生成旁白全文"""
        parts = []

        if hook.text:
            parts.append(hook.text)

        for scene in scenes:
            if scene.narration and scene.narration != hook.text:
                parts.append(scene.narration)

        return "。".join(parts) + "。"

    def generate_short(
        self,
        trend: TrendReport | dict[str, Any],
        clips: CandidateClips | dict[str, Any],
        destination: str,
        duration: int = 15,
        style: str = "fast_cut",
    ) -> Script:
        """生成旅行短视频剧本（15-20秒，高密度快剪）"""
        return self.generate(trend, clips, destination, duration)
