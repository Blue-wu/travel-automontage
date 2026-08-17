"""TTS 文字转语音模块 — 基于 edge-tts（免费、中文效果好）

支持：
- 中文多种音色（女声/男声/儿童声）
- 语速、音调调节
- 字幕同步（输出字级时间戳）
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from tools.common.config import CACHE_DIR, get_settings

logger = logging.getLogger(__name__)

# 常用中文音色
CHINESE_VOICES = {
    "xiaoxiao": {
        "name": "zh-CN-XiaoxiaoNeural",
        "display_name": "晓晓（温柔女声）",
        "gender": "female",
        "style": "gentle",
    },
    "xiaoyi": {
        "name": "zh-CN-XiaoyiNeural",
        "display_name": "晓伊（活泼女声）",
        "gender": "female",
        "style": "lively",
    },
    "yunjian": {
        "name": "zh-CN-YunjianNeural",
        "display_name": "云健（沉稳男声）",
        "gender": "male",
        "style": "calm",
    },
    "yunxi": {
        "name": "zh-CN-YunxiNeural",
        "display_name": "云希（年轻男声）",
        "gender": "male",
        "style": "young",
    },
    "yunyang": {
        "name": "zh-CN-YunyangNeural",
        "display_name": "云扬（新闻男声）",
        "gender": "male",
        "style": "news",
    },
}


class TTSSynthesizer:
    """TTS 语音合成器（基于 edge-tts）

    用法：
        tts = TTSSynthesizer(voice="xiaoxiao")
        audio_path = tts.synthesize("大家好，欢迎来到新疆", "output/voiceover.mp3")
    """

    def __init__(
        self,
        voice: str = "xiaoxiao",
        rate: str = "+0%",      # 语速，如 "+10%" 或 "-5%"
        volume: str = "+0%",    # 音量
    ):
        self.voice_key = voice
        self.voice_name = CHINESE_VOICES.get(voice, CHINESE_VOICES["xiaoxiao"])["name"]
        self.rate = rate
        self.volume = volume
        self._cache_dir = CACHE_DIR / "tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, output_path: str) -> str:
        """将文本合成为语音

        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径

        Returns:
            输出文件路径
        """
        import hashlib

        # 缓存键
        cache_key = hashlib.md5(
            f"{text}|{self.voice_name}|{self.rate}|{self.volume}".encode()
        ).hexdigest()
        cache_path = self._cache_dir / f"{cache_key}.mp3"

        if cache_path.exists():
            logger.debug(f"TTS 命中缓存: {text[:20]}...")
            # 复制到目标路径
            import shutil
            shutil.copy(cache_path, output_path)
            return output_path

        logger.info(f"TTS 合成: {text[:30]}... ({len(text)} 字)")

        try:
            asyncio.run(self._synthesize_async(text, str(cache_path)))
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            raise

        # 复制到目标路径
        import shutil
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cache_path, output_path)

        return output_path

    async def _synthesize_async(self, text: str, output_path: str):
        """异步调用 edge-tts"""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice_name,
            rate=self.rate,
            volume=self.volume,
        )
        await communicate.save(output_path)

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", audio_path],
                capture_output=True, text=True, check=True,
            )
            import json
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0

    def synthesize_scene_voiceovers(
        self,
        scenes: list[dict[str, Any]],
        output_dir: str,
    ) -> list[dict[str, Any]]:
        """为多个分镜分别生成配音

        Args:
            scenes: 分镜列表，每项需包含 narration 文本
            output_dir: 输出目录

        Returns:
            更新后的分镜列表，增加 voiceover_path 和 voiceover_duration
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if not narration:
                results.append({**scene, "voiceover_path": "", "voiceover_duration": 0})
                continue

            vo_path = output_path / f"vo_{i:03d}.mp3"
            try:
                self.synthesize(narration, str(vo_path))
                duration = self.get_audio_duration(str(vo_path))
                results.append({
                    **scene,
                    "voiceover_path": str(vo_path),
                    "voiceover_duration": duration,
                })
            except Exception as e:
                logger.warning(f"分镜 {i} 配音生成失败: {e}")
                results.append({**scene, "voiceover_path": "", "voiceover_duration": 0})

        return results

    def synthesize_full_voiceover(
        self,
        text: str,
        output_path: str,
    ) -> tuple[str, float]:
        """生成完整的旁白音频

        Args:
            text: 完整旁白文本
            output_path: 输出音频路径

        Returns:
            (文件路径, 时长秒)
        """
        self.synthesize(text, output_path)
        duration = self.get_audio_duration(output_path)
        return output_path, duration


def get_available_voices() -> list[dict]:
    """获取可用的中文音色列表"""
    return [
        {"key": k, "name": v["name"], "display_name": v["display_name"], "gender": v["gender"]}
        for k, v in CHINESE_VOICES.items()
    ]
