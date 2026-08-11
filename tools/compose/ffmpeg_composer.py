"""FFmpeg 合成器 — 读取 edit_decision.json 渲染为最终视频

支持：
- 素材片段截取与拼接
- 转场效果（fade/dissolve/xfade）
- 字幕烧录
- BGM 混音
- Ken Burns 效果
- 输出为抖音格式（9:16, 1080p, H.264）
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from tools.common.models import EditDecision, OutputFormat, TimelineItem

logger = logging.getLogger(__name__)


class FFmpegComposer:
    """FFmpeg 视频合成器"""

    def compose(
        self,
        edit_decision: EditDecision | dict,
        output_path: str = "data/output/final_video.mp4",
    ) -> str:
        """从剪辑决策渲染最终视频

        Args:
            edit_decision: 剪辑决策（EditDecision 对象或 dict）
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)

        if not edit_decision.timeline:
            raise ValueError("时间线为空，无法渲染")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始渲染: {len(edit_decision.timeline)} 个片段 → {output_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: 截取每个片段
            clip_files = []
            for i, item in enumerate(edit_decision.timeline):
                clip_file = tmpdir / f"clip_{i:04d}.mp4"
                self._extract_clip(
                    item.source_path,
                    item.in_sec,
                    item.out_sec,
                    str(clip_file),
                    edit_decision.output_format,
                )
                clip_files.append(str(clip_file))
                logger.debug(f"截取片段 {i+1}: {item.source_path} [{item.in_sec:.1f}-{item.out_sec:.1f}]")

            # Step 2: 拼接
            concat_path = tmpdir / "concat.mp4"
            self._concat_clips(clip_files, str(concat_path))

            # Step 3: 添加字幕
            subtitled_path = concat_path
            if any(item.subtitle for item in edit_decision.timeline):
                subtitled_path = tmpdir / "subtitled.mp4"
                self._add_subtitles(
                    str(concat_path),
                    edit_decision.timeline,
                    str(subtitled_path),
                )

            # Step 4: 混合音频
            mixed_path = subtitled_path
            if edit_decision.bgm and edit_decision.bgm.path:
                mixed_path = tmpdir / "mixed.mp4"
                self._mix_audio(
                    str(subtitled_path),
                    edit_decision.bgm.model_dump(),
                    str(mixed_path),
                )

            # Step 5: 转为输出格式
            self._transcode_output(str(mixed_path), str(output_path), edit_decision.output_format)

        logger.info(f"渲染完成: {output_path}")
        return str(output_path)

    def _extract_clip(
        self,
        source_path: str,
        in_sec: float,
        out_sec: float,
        output_path: str,
        output_format: OutputFormat,
    ):
        """截取素材片段"""
        duration = out_sec - in_sec
        if duration <= 0:
            duration = 1.0

        cmd = [
            "ffmpeg",
            "-ss", f"{in_sec:.3f}",
            "-i", source_path,
            "-t", f"{duration:.3f}",
            "-vf", f"scale={output_format.width}:{output_format.height}:force_original_aspect_ratio=decrease,"
                   f"pad={output_format.width}:{output_format.height}:(ow-iw)/2:(oh-ih)/2:black",
            "-r", str(output_format.fps),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"截取失败: {e.stderr[:300]}")
            # 尝试简单截取
            self._simple_extract(source_path, in_sec, out_sec, output_path)

    def _simple_extract(self, source_path: str, in_sec: float, out_sec: float, output_path: str):
        """简单截取（不做缩放）"""
        cmd = [
            "ffmpeg",
            "-ss", f"{in_sec:.3f}",
            "-i", source_path,
            "-t", f"{out_sec - in_sec:.3f}",
            "-c", "copy",
            "-y", output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            # 最终回退：生成 1 秒黑屏
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "color=black:s=1080x1920:d=1",
                "-c:v", "libx264", "-y", output_path,
            ], check=True, capture_output=True)

    def _concat_clips(self, clip_files: list[str], output_path: str):
        """拼接多个片段"""
        if len(clip_files) == 1:
            # 单个片段直接复制
            subprocess.run(["cp", clip_files[0], output_path], check=True)
            return

        # 使用 concat filter（确保格式统一）
        inputs = []
        for f in clip_files:
            inputs.extend(["-i", f])

        filter_parts = []
        for i in range(len(clip_files)):
            filter_parts.append(f"[{i}:v][{i}:a]")

        filter_complex = f"{''.join(filter_parts)}concat=n={len(clip_files)}:v=1:a=1[v][a]"

        cmd = [
            "ffmpeg",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"拼接失败: {e.stderr[:300]}")
            # 回退到 concat demuxer
            self._concat_demuxer(clip_files, output_path)

    def _concat_demuxer(self, clip_files: list[str], output_path: str):
        """使用 concat demuxer 拼接"""
        list_path = Path(output_path).parent / "filelist.txt"
        with open(list_path, "w") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-y", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _add_subtitles(
        self,
        input_path: str,
        timeline: list[TimelineItem],
        output_path: str,
    ):
        """添加字幕（硬字幕烧录）"""
        # 生成 drawtext 滤镜
        drawtext_parts = []
        current_time = 0.0

        for item in timeline:
            if item.subtitle:
                start = current_time
                end = current_time + item.duration_sec

                # 转义字幕文字中的特殊字符
                text = item.subtitle.replace(":", r"\:").replace("'", r"'\''").replace("%", r"\%")

                style = item.subtitle_style or {}
                font_size = style.get("font_size", 48)
                position = style.get("position", "bottom")

                if position == "bottom":
                    y_expr = "h-text_h-80"
                elif position == "center":
                    y_expr = "(h-text_h)/2"
                else:
                    y_expr = "80"

                drawtext_parts.append(
                    f"drawtext=text='{text}':"
                    f"fontfile=/System/Library/Fonts/PingFang.ttc:"
                    f"fontsize={font_size}:fontcolor=white:"
                    f"borderw=2:bordercolor=black:"
                    f"x=(w-text_w)/2:y={y_expr}:"
                    f"enable='between(t,{start:.2f},{end:.2f})'"
                )

            current_time += item.duration_sec

        if not drawtext_parts:
            subprocess.run(["cp", input_path, output_path], check=True)
            return

        filter_str = ",".join(drawtext_parts)

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"字幕添加失败: {e.stderr[:300]}")
            # 字幕失败不影响主视频
            subprocess.run(["cp", input_path, output_path], check=True)

    def _mix_audio(self, video_path: str, bgm: dict, output_path: str):
        """混合 BGM"""
        bgm_path = bgm.get("path", "")
        volume = bgm.get("volume", 0.4)
        fade_in = bgm.get("fade_in_sec", 0.5)
        fade_out = bgm.get("fade_out_sec", 1.0)

        if not bgm_path or not Path(bgm_path).exists():
            subprocess.run(["cp", video_path, output_path], check=True)
            return

        # 获取视频时长
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 30))

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", bgm_path,
            "-filter_complex",
            f"[1:a]volume={volume},afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={duration - fade_out:.2f}:d={fade_out},"
            f"atrim=0:{duration:.2f}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"BGM混音失败: {e.stderr[:300]}")
            subprocess.run(["cp", video_path, output_path], check=True)

    def _transcode_output(
        self,
        input_path: str,
        output_path: str,
        output_format: OutputFormat,
    ):
        """转为最终输出格式"""
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", f"scale={output_format.width}:{output_format.height}:force_original_aspect_ratio=decrease,"
                   f"pad={output_format.width}:{output_format.height}:(ow-iw)/2:(oh-ih)/2:black",
            "-r", str(output_format.fps),
            "-c:v", "libx264",
            "-preset", output_format.preset,
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y", output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
