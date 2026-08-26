"""配音生成工具 — 将剧本旁白合成为语音"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from tools.audio.tts_synthesizer import TTSSynthesizer
from tools.common.models import EditDecision, Script, VoiceoverConfig

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

    def regenerate(
        self,
        edit_decision: EditDecision | dict[str, Any],
        hook_text: str = "",
        cta_text: str = "",
        output_dir: str = "data/output/voiceover",
        voice: str | None = None,
        rate: str | None = None,
    ) -> dict[str, Any]:
        """从 edit_decision 重新生成配音（分段合成 + 精确时长 + concat 合并）

        策略（配音-字幕同步的单一真源）：
        1) 对 hook / 每段 scene / cta **分别单独做 TTS**，拿到每段的精确原声时长；
        2) 用 ffmpeg concat demuxer 把所有小段按顺序无损拼成 full_voiceover.mp3；
        3) 返回 segments 元数据：`[{kind,text,duration_sec,path}]` 与 scene_durations
           供 ffmpeg_composer 写 Bottom 字幕时使用（每段 Bottom 字幕窗口 =
           段原声时长 × 整条 atempo_ratio，保证跟读跟配音逐字对齐）。
        """
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)

        if voice or rate:
            self.tts = TTSSynthesizer(
                voice=voice if voice else "yunxi",
                rate=rate if rate else "+0%",
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        segments_dir = output_path / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        full_vo_path = output_path / "full_voiceover.mp3"

        # Step 1: 收集待合成的文本（附带 kind 标记，方便后续映射）
        tagged: list[tuple[str, str]] = []  # (kind, text)
        if hook_text:
            tagged.append(("hook", hook_text))
        for item in edit_decision.timeline:
            if item.narration_text:
                tagged.append(("scene", item.narration_text))
        if cta_text:
            tagged.append(("cta", cta_text))

        if not tagged:
            logger.warning("无旁白文本，跳过配音重新生成")
            return {"path": "", "volume": 1.0, "duration": 0,
                    "scene_durations": [], "segments": []}

        # Step 2: 每段单独 TTS，拿到精确原声时长
        segments: list[dict[str, Any]] = []
        scene_durations: list[float] = []
        total_duration = 0.0
        seg_paths: list[str] = []
        logger.info(f"重新生成配音: {len(tagged)} 段 (hook={sum(1 for t,_ in tagged if t=='hook')},"
                    f" scene={sum(1 for t,_ in tagged if t=='scene')},"
                    f" cta={sum(1 for t,_ in tagged if t=='cta')})")

        for idx, (kind, text) in enumerate(tagged):
            seg_path = str(segments_dir / f"seg_{idx:03d}_{kind}.mp3")
            try:
                self.tts.synthesize(text, seg_path)
                dur = self.tts.get_audio_duration(seg_path)
                if dur <= 0:
                    # 兜底：用字数估 0.28s/字（中文约 3.5 字/秒）
                    dur = max(0.4, len(text) * 0.28)
            except Exception as e:
                logger.warning(f"配音段 {idx} [{kind}] 合成失败: {e}，用字数兜底")
                dur = max(0.4, len(text) * 0.28)
            total_duration += dur
            seg_info = {
                "kind": kind,
                "text": text,
                "duration_sec": round(dur, 4),
                "path": seg_path,
            }
            segments.append(seg_info)
            seg_paths.append(seg_path)
            if kind == "scene":
                scene_durations.append(round(dur, 4))

        # Step 3: 用 ffmpeg concat demuxer 按顺序无损拼接成整条配音
        if len(seg_paths) == 1:
            # 只有一段，直接复制
            import shutil as _shutil
            _shutil.copy(seg_paths[0], str(full_vo_path))
        else:
            list_file = output_path / "segments_concat.txt"
            with open(list_file, "w", encoding="utf-8") as lf:
                for p in seg_paths:
                    # concat demuxer 要求单引号路径、内部单引号转义为 '\''
                    escaped = p.replace("'", "'\\''")
                    lf.write(f"file '{escaped}'\n")
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy", "-y", str(full_vo_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                # concat demuxer 失败 → 用 re-encode 兜底
                logger.warning(f"concat demuxer 失败，回退重编码拼接: {e.stderr[:200]}")
                inputs = []
                for p in seg_paths:
                    inputs += ["-i", p]
                filter_parts = "".join(f"[{i}:a]" for i in range(len(seg_paths)))
                cmd2 = [
                    "ffmpeg", *inputs,
                    "-filter_complex",
                    f"{filter_parts}concat=n={len(seg_paths)}:v=0:a=1[outa]",
                    "-map", "[outa]", "-c:a", "libmp3lame", "-b:a", "192k",
                    "-y", str(full_vo_path),
                ]
                subprocess.run(cmd2, check=True, capture_output=True, text=True)

        # 重新测一下整条时长（ffprobe 更准），兜底用总和
        final_duration = self.tts.get_audio_duration(str(full_vo_path))
        if final_duration <= 0:
            final_duration = total_duration

        logger.info(
            f"配音重新生成完成: 共 {len(segments)} 段, "
            f"原声总时长 {final_duration:.1f}秒 "
            f"(scene 段 {len(scene_durations)} 条)"
        )

        return {
            "path": str(full_vo_path),
            "volume": 1.0,
            "duration": round(final_duration, 4),
            "scene_durations": scene_durations,
            "segments": segments,
        }
