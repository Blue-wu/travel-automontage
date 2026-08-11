"""分镜规划器 — 将剧本拆解为分镜表，匹配具体素材

Agent 在 storyboard 阶段调用此工具，
同时读取 skills/travel-pacing.md 获取节奏规范。
"""

from __future__ import annotations

import logging
from typing import Any

from tools.common.models import CandidateClips, Script, ScriptScene

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

        storyboard = []
        for i, scene in enumerate(script_scenes):
            # 匹配素材
            matched_clip = self._match_clip(scene, clips, i)

            storyboard.append({
                "order": i + 1,
                "script_scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                "matched_clip": matched_clip.model_dump() if hasattr(matched_clip, "model_dump") else matched_clip,
                "shot_type": self._suggest_shot_type(scene, i),
                "transition": scene.transition if hasattr(scene, "transition") else scene.get("transition", "cut"),
            })

        logger.info(f"分镜规划完成: {len(storyboard)} 个分镜")
        return storyboard

    def plan_beat_aligned(
        self,
        candidate_clips: CandidateClips | dict[str, Any],
        bgm: dict[str, Any],
        duration: int = 45,
        bpm: int = 120,
    ) -> list[dict[str, Any]]:
        """节拍对齐分镜（用于混剪）

        将素材片段对齐到 BGM 节拍点。
        """
        logger.info(f"节拍对齐分镜: bpm={bpm}, duration={duration}s")

        clips = self._extract_clips(candidate_clips)

        # 计算节拍点
        beat_interval = 60.0 / bpm  # 每拍秒数
        total_beats = int(duration / beat_interval)

        # 按节拍分配素材
        storyboard = []
        clip_idx = 0
        current_time = 0.0

        for beat in range(total_beats):
            if current_time >= duration:
                break

            # 每 2-4 拍换一个镜头
            shots_per_clip = 2 if beat % 4 < 2 else 3
            shot_duration = beat_interval * shots_per_clip

            if clip_idx < len(clips):
                clip = clips[clip_idx]
                clip_duration = clip.end_sec - clip.start_sec if hasattr(clip, "end_sec") else \
                    clip.get("end_sec", 0) - clip.get("start_sec", 0)

                # 如果素材够长，截取一段；否则用整段
                if clip_duration >= shot_duration:
                    in_sec = clip.start_sec if hasattr(clip, "start_sec") else clip.get("start_sec", 0)
                    out_sec = in_sec + shot_duration
                else:
                    in_sec = clip.start_sec if hasattr(clip, "start_sec") else clip.get("start_sec", 0)
                    out_sec = clip.end_sec if hasattr(clip, "end_sec") else clip.get("end_sec", shot_duration)

                storyboard.append({
                    "order": beat + 1,
                    "start_time": current_time,
                    "duration_sec": shot_duration,
                    "matched_clip": clip.model_dump() if hasattr(clip, "model_dump") else clip,
                    "transition": "cut",
                    "on_beat": True,
                })

                clip_idx += 1
            else:
                # 循环使用素材
                clip_idx = 0

            current_time += shot_duration

        return storyboard

    def plan_fast_cut(
        self,
        script: Script | dict[str, Any],
        candidate_clips: CandidateClips | dict[str, Any],
        avg_shot_duration: float = 1.5,
    ) -> list[dict[str, Any]]:
        """快剪分镜（用于短视频）"""
        logger.info(f"快剪分镜: avg_shot={avg_shot_duration}s")

        script_scenes = self._extract_script_scenes(script)
        clips = self._extract_clips(candidate_clips)

        storyboard = []
        clip_idx = 0

        for i, scene in enumerate(script_scenes):
            # 短视频每镜 1-2 秒
            num_shots = max(1, int(scene.duration_sec / avg_shot_duration))

            for j in range(num_shots):
                shot_duration = scene.duration_sec / num_shots
                matched_clip = clips[clip_idx % len(clips)] if clips else None

                storyboard.append({
                    "order": len(storyboard) + 1,
                    "script_scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                    "matched_clip": matched_clip.model_dump() if matched_clip and hasattr(matched_clip, "model_dump") else matched_clip,
                    "shot_type": "close_up" if j % 2 == 0 else "wide",
                    "transition": "cut",
                    "duration_sec": shot_duration,
                })
                clip_idx += 1

        return storyboard

    def _extract_script_scenes(self, script: Any) -> list[ScriptScene]:
        """从 Script 对象或 dict 中提取场景列表"""
        if isinstance(script, Script):
            return script.scenes
        elif isinstance(script, dict):
            scenes_data = script.get("scenes", [])
            return [ScriptScene(**s) if isinstance(s, dict) else s for s in scenes_data]
        return []

    def _extract_clips(self, clips: Any) -> list:
        """从 CandidateClips 对象或 dict 中提取片段列表"""
        if isinstance(clips, CandidateClips):
            return clips.clips
        elif isinstance(clips, dict):
            return clips.get("clips", [])
        return []

    def _match_clip(self, scene: ScriptScene, clips: list, index: int) -> Any:
        """为分镜匹配素材"""
        if not clips:
            return None

        # 简单策略：轮转使用
        return clips[index % len(clips)]

    def _suggest_shot_type(self, scene: ScriptScene, index: int) -> str:
        """建议景别"""
        phase = scene.mood if hasattr(scene, "mood") else scene.get("mood", "")

        if phase in ("shock", "epic"):
            return "extreme_wide"  # 大远景
        elif phase in ("lively", "energetic"):
            return "close_up"  # 近景/特写
        elif phase in ("curious", "moving"):
            return "medium"  # 中景
        elif phase in ("calm",):
            return "wide"  # 远景
        else:
            return "medium"
