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
        "style": "gentle",     # 仅用于选音色，edge-tts 不下发
        "rate": "+8%", "pitch": "+0Hz",
    },
    "xiaoyi": {
        "name": "zh-CN-XiaoyiNeural",
        "display_name": "晓伊（活泼女声）",
        "gender": "female",
        "style": "lively",
        "rate": "+14%", "pitch": "+3Hz",
    },
    "yunjian": {
        "name": "zh-CN-YunjianNeural",
        "display_name": "云健（沉稳男声）",
        "gender": "male",
        "style": "calm",
        "rate": "+4%", "pitch": "+0Hz",
    },
    "yunxi": {
        "name": "zh-CN-YunxiNeural",
        "display_name": "云希（年轻男声）",
        "gender": "male",
        "style": "young",
        "rate": "+12%", "pitch": "+2Hz",
    },
    "yunyang": {
        "name": "zh-CN-YunyangNeural",
        "display_name": "云扬（新闻男声）",
        "gender": "male",
        "style": "news",
        "rate": "+0%", "pitch": "+0Hz",   # 播音腔，旅行 vlog 不建议
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
        rate: str | None = None,     # 语速，如 "+10%"；None = 用音色的推荐值
        volume: str = "+0%",
        pitch: str | None = None,    # 音调，如 "+3Hz"；None = 用音色的推荐值
    ):
        cfg = CHINESE_VOICES.get(voice, CHINESE_VOICES["xiaoxiao"])
        self.voice_key = voice
        self.voice_name = cfg["name"]
        # 默认 +0% 是播音腔。旅行 vlog 稍快、音调略高更像真人在说话。
        # ⚠️ edge-tts 只支持 rate / pitch / volume —— 它是 Edge 朗读服务的封装，
        #    mstts:express-as 那套情感标签属于 Azure Speech 正式服务，传了也无效。
        #    CHINESE_VOICES 里的 style 字段仅用于选音色，不下发给引擎。
        #    真要情感控制得换 CosyVoice（DashScope，付费）。
        self.rate = rate if rate is not None else cfg.get("rate", "+8%")
        self.pitch = pitch if pitch is not None else cfg.get("pitch", "+0Hz")
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
            f"{text}|{self.voice_name}|{self.rate}|{self.pitch}|{self.volume}".encode()
        ).hexdigest()
        cache_path = self._cache_dir / f"{cache_key}.mp3"

        if cache_path.exists():
            logger.debug(f"TTS 命中缓存: {text[:20]}...")
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache_path, output_path)
            src_t = self._timings_path(cache_path)
            if src_t.exists():
                shutil.copy(src_t, self._timings_path(output_path))
            return output_path

        logger.info(f"TTS 合成: {text[:30]}... ({len(text)} 字)")

        try:
            timings = asyncio.run(self._synthesize_async(text, str(cache_path)))
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            raise

        # 字级时间戳与音频同名落盘，缓存命中时一并复用
        if timings:
            import json
            self._timings_path(cache_path).write_text(
                json.dumps(timings, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"  字级时间戳 {len(timings)} 条")

        # 复制到目标路径
        import shutil
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cache_path, output_path)
        src_t = self._timings_path(cache_path)
        if src_t.exists():
            shutil.copy(src_t, self._timings_path(output_path))

        return output_path

    async def _synthesize_async(self, text: str, output_path: str) -> list[dict]:
        """异步调用 edge-tts，同时收集字级时间戳

        原实现用 communicate.save()，事件流被整个丢掉 —— 所以文件头声称的
        「输出字级时间戳」一直没有实现。改用 stream() 逐块消费，
        WordBoundary 事件给出每个词/字的 offset 和 duration（单位 100 纳秒）。

        这些时间戳用来驱动 ASS 的 \\k 标签，做逐字点亮（见
        tools/compose/subtitle_style.py），字幕才能真正跟着人声走。
        """
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice_name,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        timings: list[dict] = []
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                ctype = chunk.get("type")
                if ctype == "audio":
                    f.write(chunk["data"])
                elif ctype == "WordBoundary":
                    # offset / duration 单位是 100ns tick
                    start = chunk.get("offset", 0) / 1e7
                    dur = chunk.get("duration", 0) / 1e7
                    timings.append({
                        "text": chunk.get("text", ""),
                        "start": round(start, 3),
                        "end": round(start + dur, 3),
                    })
        return timings

    @staticmethod
    def _timings_path(audio_path: str | Path) -> Path:
        return Path(str(audio_path).rsplit(".", 1)[0] + ".timings.json")

    def get_word_timings(self, audio_path: str) -> list[tuple[str, float, float]]:
        """读取与音频同名的字级时间戳，供 ASS 逐字点亮使用

        Returns: [(文本, 起, 止)]，没有则返回空列表（降级为整句淡入）
        """
        import json

        tp = self._timings_path(audio_path)
        if not tp.exists():
            return []
        try:
            data = json.loads(tp.read_text(encoding="utf-8"))
            return [(d["text"], float(d["start"]), float(d["end"])) for d in data]
        except Exception as e:
            logger.debug(f"读取时间戳失败 {tp}: {e}")
            return []

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
