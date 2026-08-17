"""配音生成工具 — 将剧本旁白合成为语音"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tools.audio.tts_synthesizer import TTSSynthesizer
from tools.common.models import Script, VoiceoverConfig

logger = logging.getLogger(__name__)


class VoiceoverGenerator:
    """旁白配音生成器

    将剧本中的旁白文本合成为语音，
    并生成配音配置供视频合成使用。
    """

    def __init__(self, voice: str = "yunxi", rate: str = "+0%"):
        self.tts = TTSSynthesizer(voice=voice, rate=rate)

    def generate(
        self,
        script: Script | dict[str, Any],
        output_dir: str = "data/output/voiceover",
        voice: str | None = None,
        rate: str | None = None,
    ) -> dict[str, Any]:
        """从剧本生成完整旁白配音

        策略：
        1. 把所有分镜的旁白连起来（中间加短停顿）
        2. 生成完整的配音 mp3
        3. 返回配音配置

        Args:
            script: 剧本对象或 dict
            output_dir: 输出目录

        Returns:
            {
                "path": "配音文件路径",
                "volume": 1.0,
                "duration": 12.5,
                "scene_durations": [...]  # 每个分镜配音时长
            }
        """
        if isinstance(script, dict):
            script = Script(**script)

        # 如果指定了 voice/rate，重新初始化 TTS
        if voice or rate:
            self.tts = TTSSynthesizer(
                voice=voice if voice else "yunxi",
                rate=rate if rate else "+0%",
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        full_vo_path = output_path / "full_voiceover.mp3"

        # 收集所有旁白文本
        narrations = []

        # 钩子
        if script.hook:
            hook_text = script.hook.text if hasattr(script.hook, 'text') else str(script.hook)
            if hook_text:
                narrations.append(hook_text)

        # 分镜旁白
        for scene in script.scenes:
            if scene.narration:
                narrations.append(scene.narration)

        # CTA
        if script.cta:
            cta_text = script.cta.text if hasattr(script.cta, 'text') else str(script.cta)
            if cta_text:
                narrations.append(cta_text)

        if not narrations:
            logger.warning("剧本中没有旁白文本，跳过配音生成")
            return {
                "path": "",
                "volume": 1.0,
                "duration": 0,
                "scene_durations": [],
            }

        # 用停顿连接所有旁白
        full_text = "。".join(narrations)
        logger.info(f"生成配音: {len(narrations)} 段, 共 {len(full_text)} 字")

        try:
            self.tts.synthesize(full_text, str(full_vo_path))
            duration = self.tts.get_audio_duration(str(full_vo_path))
            logger.info(f"配音生成完成: {duration:.1f}秒")
        except Exception as e:
            logger.error(f"配音生成失败: {e}")
            return {
                "path": "",
                "volume": 1.0,
                "duration": 0,
                "scene_durations": [],
            }

        return {
            "path": str(full_vo_path),
            "volume": 1.0,
            "duration": duration,
            "scene_durations": [],
        }
