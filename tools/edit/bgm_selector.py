"""BGM 选择器 — 根据 BGM 趋势和目标情绪选择音乐"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BGMSelector:
    """BGM 选择器"""

    # 情绪到 BPM 的映射
    MOOD_BPM = {
        "chill": (60, 80),
        "energetic": (100, 130),
        "epic": (80, 120),
        "emotional": (70, 90),
        "fast": (120, 140),
    }

    def select(
        self,
        mood: str = "chill",
        trend_music: list[dict[str, Any]] | None = None,
        duration: int = 30,
    ) -> dict[str, Any]:
        """选择 BGM

        Args:
            mood: 目标情绪
            trend_music: 趋势分析输出的音乐趋势
            duration: 视频时长

        Returns:
            BGM 配置 dict
        """
        logger.info(f"选择 BGM: mood={mood}, duration={duration}s")

        # 从趋势音乐中匹配
        if trend_music:
            for music in trend_music:
                music_mood = music.get("mood", "")
                if mood.lower() in music_mood.lower() or music_mood.lower() in mood.lower():
                    bpm_range = self.MOOD_BPM.get(mood, (80, 120))
                    bpm = (bpm_range[0] + bpm_range[1]) // 2
                    return {
                        "name": music.get("bgm_name", ""),
                        "mood": mood,
                        "bpm": bpm,
                        "path": "",  # 需要用户提供实际 BGM 文件路径
                        "volume": 0.4,
                        "fade_in_sec": 0.5,
                        "fade_out_sec": 1.0,
                    }

        # 回退：返回默认配置
        bpm_range = self.MOOD_BPM.get(mood, (80, 120))
        bpm = (bpm_range[0] + bpm_range[1]) // 2

        return {
            "name": "",
            "mood": mood,
            "bpm": bpm,
            "path": "",
            "volume": 0.4,
            "fade_in_sec": 0.5,
            "fade_out_sec": 1.0,
        }
