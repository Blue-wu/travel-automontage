"""分镜规划器 — 将剧本拆解为分镜表，匹配具体素材

Agent 在 storyboard 阶段调用此工具，
同时读取 skills/travel-pacing.md 获取节奏规范。

严格去重规则：
- 同一个视频源文件最多使用 2 次
- 相邻分镜绝对不能使用同一段素材
- 优先使用未被使用过的素材
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tools.common.models import CandidateClip, CandidateClips, Script, ScriptScene, SceneCategory

logger = logging.getLogger(__name__)


class StoryboardPlanner:
    """分镜规划器"""

    def plan(
        self,
        script: Script | dict[str, Any],
        candidate_clips: CandidateClips | dict[str, Any],
    ) -> list[dict[str, Any]]:
        """将剧本拆解为分镜表

        Returns:
            分镜列表，每个分镜包含：
            - script_scene: 剧本中的分镜信息
            - matched_clip: 匹配的素材片段
        """
        logger.info("开始分镜规划")

        script_scenes = self._extract_script_scenes(script)
        clips = self._extract_clips(candidate_clips)
        categories = self._extract_categories(candidate_clips)

        category_map = {c.category: c.clips for c in categories} if categories else {}

        used_source_files: dict[str, int] = {}
        last_clip_key: str | None = None
        storyboard = []

        for i, scene in enumerate(script_scenes):
            matched_clip, clip_key = self._match_clip(
                scene,
                clips,
                category_map,
                used_source_files,
                last_clip_key,
                i,
            )

            if matched_clip is not None:
                source_file = self._get_source_key(matched_clip)
                used_source_files[source_file] = used_source_files.get(source_file, 0) + 1
                last_clip_key = clip_key

            storyboard.append({
                "order": i + 1,
                "script_scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                "matched_clip": matched_clip.model_dump() if hasattr(matched_clip, "model_dump") else matched_clip,
                "shot_type": scene.shot_type if hasattr(scene, "shot_type") else "medium",
                "transition": scene.transition if hasattr(scene, "transition") else "cut",
            })

        unique_sources = len(used_source_files)
        total_clips = len(storyboard)
        reuse_count = total_clips - unique_sources
        logger.info(f"分镜规划完成: {total_clips} 个分镜, 使用 {unique_sources} 个不同素材, 重复 {reuse_count} 次")

        return storyboard

    def _match_clip(
        self,
        scene: ScriptScene,
        clips: list[CandidateClip],
        category_map: dict[str, list[CandidateClip]],
        used_source_files: dict[str, int],
        last_clip_key: str | None,
        scene_index: int,
    ) -> tuple[CandidateClip | None, str | None]:
        """为分镜匹配素材 — 基于关键词 + 类别 + 去重约束"""
        if not clips:
            return None, None

        visual_keywords = self._get_visual_keywords(scene)
        preferred_categories = self._get_preferred_categories(scene)

        candidates_pool = self._get_candidate_pool(preferred_categories, category_map, clips)

        scored = []
        for clip in candidates_pool:
            clip_key = self._get_clip_key(clip)
            source_key = self._get_source_key(clip)

            score = self._compute_match_score(visual_keywords, clip)

            use_count = used_source_files.get(source_key, 0)
            if use_count >= 2:
                score -= 10.0
            elif use_count == 1:
                score -= 0.5

            if clip_key == last_clip_key:
                score -= 100.0

            if hasattr(clip, "quality"):
                score += clip.quality * 0.3

            scored.append((clip, clip_key, score))

        scored.sort(key=lambda x: -x[2])

        if not scored:
            return None, None

        best_clip, best_key, best_score = scored[0]

        if best_score < -5.0 and len(clips) > len(used_source_files):
            fallback = self._find_least_used_clip(clips, used_source_files, last_clip_key)
            if fallback:
                return fallback, self._get_clip_key(fallback)

        return best_clip, best_key

    def _get_candidate_pool(
        self,
        preferred_categories: list[str],
        category_map: dict[str, list[CandidateClip]],
        all_clips: list[CandidateClip],
    ) -> list[CandidateClip]:
        """获取候选池 — 优先从偏好类别里选"""
        if not preferred_categories:
            return all_clips

        pool = []
        for cat in preferred_categories:
            if cat in category_map:
                pool.extend(category_map[cat])

        if pool:
            return pool

        return all_clips

    def _find_least_used_clip(
        self,
        clips: list[CandidateClip],
        used_source_files: dict[str, int],
        last_clip_key: str | None,
    ) -> CandidateClip | None:
        """找使用次数最少的素材（兜底用）"""
        candidates = []
        for clip in clips:
            clip_key = self._get_clip_key(clip)
            if clip_key == last_clip_key:
                continue

            source_key = self._get_source_key(clip)
            use_count = used_source_files.get(source_key, 0)
            quality = getattr(clip, "quality", 0.5)

            candidates.append((clip, use_count, quality))

        if not candidates:
            return clips[0] if clips else None

        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[0][0]

    def _compute_match_score(self, keywords: list[str], clip: CandidateClip) -> float:
        """计算关键词与素材的匹配分数"""
        if not keywords:
            return 0.0

        clip_text = ""
        clip_tags = []

        if hasattr(clip, "scene_summary"):
            clip_text = clip.scene_summary or ""
        elif isinstance(clip, dict):
            clip_text = clip.get("scene_summary", "")

        if hasattr(clip, "visual_tags"):
            clip_tags = clip.visual_tags or []
        elif isinstance(clip, dict):
            clip_tags = clip.get("visual_tags", [])

        clip_text_lower = clip_text.lower()
        clip_tags_lower = [t.lower() for t in clip_tags]

        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()

            if kw_lower in clip_text_lower:
                score += 3.0

            for tag in clip_tags_lower:
                if kw_lower in tag or tag in kw_lower:
                    score += 2.0
                    break

        return score

    def _get_visual_keywords(self, scene: ScriptScene | dict) -> list[str]:
        """获取分镜的视觉关键词"""
        if hasattr(scene, "visual_keywords") and scene.visual_keywords:
            return list(scene.visual_keywords)
        if isinstance(scene, dict):
            return scene.get("visual_keywords", [])
        return []

    def _get_preferred_categories(self, scene: ScriptScene | dict) -> list[str]:
        """获取分镜偏好的素材类别（从视觉关键词推导）"""
        keywords = self._get_visual_keywords(scene)
        if not keywords:
            return []

        keyword_to_category = {
            "雪山": "mountain", "山": "mountain", "峰": "mountain", "冰川": "mountain",
            "湖": "lake", "湖泊": "lake", "水": "lake", "河流": "lake", "海": "lake",
            "草原": "grassland", "草地": "grassland", "牛": "grassland", "羊": "grassland",
            "森林": "forest", "树": "forest", "云杉": "forest", "林": "forest",
            "公路": "road", "路": "road", "开车": "road", "沿途": "road",
            "日落": "sunset", "夕阳": "sunset", "黄昏": "sunset", "晚霞": "sunset",
            "星空": "night", "夜景": "night", "星星": "night", "夜空": "night",
            "天空": "sky", "云": "sky", "航拍": "sky", "全景": "sky",
            "建筑": "architecture", "人文": "architecture", "街道": "architecture",
            "美食": "food", "吃": "food", "小吃": "food",
        }

        categories = []
        for kw in keywords:
            for kw_map, cat in keyword_to_category.items():
                if kw_map in kw and cat not in categories:
                    categories.append(cat)

        return categories

    @staticmethod
    def _get_source_key(clip: CandidateClip | dict) -> str:
        """获取素材的源文件标识（用于去重）"""
        if hasattr(clip, "source_path"):
            return os.path.basename(clip.source_path)
        if isinstance(clip, dict):
            path = clip.get("source_path", "")
            return os.path.basename(path)
        return str(id(clip))

    @staticmethod
    def _get_clip_key(clip: CandidateClip | dict) -> str:
        """获取片段唯一标识（源文件+时间段）"""
        source = StoryboardPlanner._get_source_key(clip)

        if hasattr(clip, "start_sec") and hasattr(clip, "end_sec"):
            return f"{source}_{clip.start_sec}_{clip.end_sec}"
        if isinstance(clip, dict):
            return f"{source}_{clip.get('start_sec', 0)}_{clip.get('end_sec', 0)}"

        return source

    def _extract_script_scenes(self, script: Any) -> list[ScriptScene]:
        """从 Script 对象或 dict 中提取场景列表"""
        if isinstance(script, Script):
            return script.scenes
        elif isinstance(script, dict):
            scenes_data = script.get("scenes", [])
            return [ScriptScene(**s) if isinstance(s, dict) else s for s in scenes_data]
        return []

    def _extract_clips(self, clips: Any) -> list[CandidateClip]:
        """从 CandidateClips 对象或 dict 中提取片段列表"""
        if isinstance(clips, CandidateClips):
            return clips.clips
        elif isinstance(clips, dict):
            clip_data = clips.get("clips", [])
            return [CandidateClip(**c) if isinstance(c, dict) else c for c in clip_data]
        return []

    def _extract_categories(self, clips: Any) -> list[SceneCategory]:
        """从 CandidateClips 中提取分类"""
        if isinstance(clips, CandidateClips):
            return clips.categories
        elif isinstance(clips, dict):
            cats = clips.get("categories", [])
            return [SceneCategory(**c) if isinstance(c, dict) else c for c in cats]
        return []
