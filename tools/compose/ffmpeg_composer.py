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
import shutil
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

        # 探测第一个素材的分辨率（如果 output_format 为 0 则保持原分辨率）
        fmt = edit_decision.output_format
        original_res_mode = fmt.width <= 0 or fmt.height <= 0
        if original_res_mode:
            first_clip = edit_decision.timeline[0].source_path
            probe_w, probe_h, probe_fps = self._probe_video_size(first_clip)
            if probe_w > 0 and probe_h > 0:
                fmt = OutputFormat(
                    width=int(probe_w),
                    height=int(probe_h),
                    fps=float(probe_fps) if probe_fps > 0 else fmt.fps,
                    codec=fmt.codec,
                    preset=fmt.preset,
                )
                edit_decision.output_format = fmt
                logger.info(f"自动适配原素材分辨率: {probe_w}x{probe_h} @ {probe_fps:.2f}fps")
        keep_original = original_res_mode

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
                    effects=item.effects,
                    clip_index=i,
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
                    edit_decision.output_format,
                )

            # Step 4: 添加旁白配音（如果有）
            voiced_path = subtitled_path
            if edit_decision.voiceover and edit_decision.voiceover.path:
                voiced_path = tmpdir / "voiced.mp4"
                self._add_voiceover(
                    str(subtitled_path),
                    edit_decision.voiceover.model_dump() if hasattr(edit_decision.voiceover, 'model_dump') else edit_decision.voiceover,
                    str(voiced_path),
                )

            # Step 5: 混合 BGM
            mixed_path = voiced_path
            if edit_decision.bgm and edit_decision.bgm.path:
                mixed_path = tmpdir / "mixed.mp4"
                self._mix_audio(
                    str(voiced_path),
                    edit_decision.bgm.model_dump() if hasattr(edit_decision.bgm, 'model_dump') else edit_decision.bgm,
                    str(mixed_path),
                )

            # Step 6: 转为输出格式（如果分辨率和帧率跟源素材一致，直接复制，跳过转码）
            if not keep_original:
                self._transcode_output(str(mixed_path), str(output_path), edit_decision.output_format)
            else:
                # 保持原分辨率：前面步骤已经是正确格式，直接复制
                import shutil as _shutil
                _shutil.copy(str(mixed_path), str(output_path))

        logger.info(f"渲染完成: {output_path}")
        return str(output_path)

    def _extract_clip(
        self,
        source_path: str,
        in_sec: float,
        out_sec: float,
        output_path: str,
        output_format: OutputFormat,
        effects: list | None = None,
        clip_index: int = 0,
    ):
        """截取素材片段"""
        duration = out_sec - in_sec
        if duration <= 0:
            duration = 1.0

        # 检查是否保持原分辨率（width <= 0 或 height <= 0 表示保持原尺寸）
        keep_original = output_format.width <= 0 or output_format.height <= 0

        vf_parts = []

        if not keep_original:
            # 基础缩放和填充
            vf_parts.append(
                f"scale={output_format.width}:{output_format.height}:force_original_aspect_ratio=decrease,"
                f"pad={output_format.width}:{output_format.height}:(ow-iw)/2:(oh-ih)/2:black"
            )

        # Vignette 暗角效果（高潮场景）
        if effects:
            for eff in effects:
                if eff.get("type") == "vignette":
                    intensity = eff.get("intensity", 0.4)
                    vf_parts.append(f"vignette={intensity}")
                    break

        vf_str = ",".join(vf_parts) if vf_parts else None

        cmd = [
            "ffmpeg",
            "-ss", f"{in_sec:.3f}",
            "-i", source_path,
            "-t", f"{duration:.3f}",
        ]
        if vf_str:
            cmd += ["-vf", vf_str, "-r", str(output_format.fps)]
        cmd += [
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
            logger.warning(f"截取失败，回退简单截取: {e.stderr[:200]}")
            self._simple_extract_with_scale(source_path, in_sec, out_sec, output_path, output_format)

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

    def _simple_extract_with_scale(
        self,
        source_path: str,
        in_sec: float,
        out_sec: float,
        output_path: str,
        output_format: OutputFormat,
    ):
        """带缩放的简单截取（特效失败时回退）"""
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
        except subprocess.CalledProcessError:
            self._simple_extract(source_path, in_sec, out_sec, output_path)

    def _concat_clips(self, clip_files: list[str], output_path: str):
        """拼接多个片段"""
        if len(clip_files) == 1:
            # 单个片段直接复制
            shutil.copy(clip_files[0], output_path)
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
        output_format: OutputFormat | None = None,
    ):
        """添加字幕（硬字幕烧录） — 使用 ASS 字幕文件，兼容性更好"""
        if not any(item.subtitle for item in timeline):
            shutil.copy(input_path, output_path)
            return

        font_path = self._get_font_path()
        ass_content = self._generate_ass_subtitles(timeline, font_path, output_format)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False, encoding="utf-8") as f:
            f.write(ass_content)
            ass_path = f.name

        try:
            ass_path_escaped = ass_path.replace("\\", "/").replace(":", r"\:")
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"ass='{ass_path_escaped}'",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-y", output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"字幕添加失败: {e.stderr[:500]}")
            try:
                self._add_subtitles_drawtext(input_path, timeline, output_path)
            except Exception as e2:
                logger.error(f"drawtext 字幕也失败: {e2}")
                shutil.copy(input_path, output_path)
        finally:
            try:
                import os
                os.unlink(ass_path)
            except Exception:
                pass

    def _generate_ass_subtitles(
        self,
        timeline: list[TimelineItem],
        font_path: str,
        output_format: OutputFormat | None = None,
    ) -> str:
        """生成 ASS 格式字幕"""
        # 基准分辨率（1080x1920 竖屏）
        base_w, base_h = 1080, 1920

        if output_format and output_format.width > 0 and output_format.height > 0:
            play_w, play_h = output_format.width, output_format.height
        else:
            play_w, play_h = base_w, base_h

        # 字体大小按高度比例缩放
        scale = play_h / base_h
        base_font_size = 48

        current_time = 0.0
        events = []

        for item in timeline:
            if not item.subtitle:
                current_time += item.duration_sec
                continue

            start = current_time
            end = current_time + item.duration_sec

            start_str = self._seconds_to_ass_time(start)
            end_str = self._seconds_to_ass_time(end)

            style = item.subtitle_style or {}
            font_size = int(style.get("font_size", base_font_size) * scale)

            events.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{item.subtitle}"
            )

            current_time += item.duration_sec

        font_name = self._get_font_name(font_path)
        outline = max(2, int(3 * scale))
        margin_v = int(80 * scale)

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        return ass_header + "\n".join(events) + "\n"

    @staticmethod
    def _seconds_to_ass_time(seconds: float) -> str:
        """秒数转 ASS 时间格式 H:MM:SS.cc"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds - int(seconds)) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    @staticmethod
    def _get_font_name(font_path: str) -> str:
        """从字体路径获取字体名（简化版）"""
        import os

        name_map = {
            "msyh.ttc": "Microsoft YaHei",
            "msyhbd.ttc": "Microsoft YaHei",
            "simhei.ttf": "SimHei",
            "simsun.ttc": "SimSun",
        }

        basename = os.path.basename(font_path)
        if basename in name_map:
            return name_map[basename]

        name, _ = os.path.splitext(basename)
        return name or "Arial"

    def _add_subtitles_drawtext(
        self,
        input_path: str,
        timeline: list[TimelineItem],
        output_path: str,
    ):
        """备用：drawtext 方式添加字幕"""
        drawtext_parts = []
        current_time = 0.0

        font_path = self._get_font_path()
        import os
        font_path_escaped = font_path.replace("\\", "/").replace(":", r"\:")

        for item in timeline:
            if item.subtitle:
                start = current_time
                end = current_time + item.duration_sec

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
                    f"fontfile='{font_path_escaped}':"
                    f"fontsize={font_size}:fontcolor=white:"
                    f"borderw=2:bordercolor=black:"
                    f"x=(w-text_w)/2:y={y_expr}:"
                    f"enable='between(t,{start:.2f},{end:.2f})'"
                )

            current_time += item.duration_sec

        if not drawtext_parts:
            shutil.copy(input_path, output_path)
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

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _add_voiceover(self, video_path: str, voiceover: dict, output_path: str):
        """添加旁白配音

        旁白音量比 BGM 大，确保听清楚。
        如果原视频有声音，会压低原声音量。
        """
        vo_path = voiceover.get("path", "")
        volume = voiceover.get("volume", 1.0)

        if not vo_path or not Path(vo_path).exists():
            shutil.copy(video_path, output_path)
            return

        # 获取视频时长
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 30))

        # 获取配音时长
        vo_probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", vo_path],
            capture_output=True, text=True, check=True,
        )
        vo_duration = float(json.loads(vo_probe.stdout).get("format", {}).get("duration", 0))

        if vo_duration <= 0:
            shutil.copy(video_path, output_path)
            return

        # 原视频声音压低到 0.3，旁白 1.0
        # 混音：原音（小声） + 旁白（大声）
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", vo_path,
            "-filter_complex",
            f"[0:a]volume=0.3[original];"
            f"[1:a]volume={volume},atrim=0:{duration:.2f}[vo];"
            f"[original][vo]amix=inputs=2:duration=first:dropout_transition=0[a]",
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
            logger.info(f"旁白配音已添加: {vo_duration:.1f}s / {duration:.1f}s")
        except subprocess.CalledProcessError as e:
            logger.error(f"旁白配音失败: {e.stderr[:300]}")
            shutil.copy(video_path, output_path)

    def _mix_audio(self, video_path: str, bgm: dict, output_path: str):
        """混合 BGM"""
        bgm_path = bgm.get("path", "")
        volume = bgm.get("volume", 0.4)
        fade_in = bgm.get("fade_in_sec", 0.5)
        fade_out = bgm.get("fade_out_sec", 1.0)

        if not bgm_path or not Path(bgm_path).exists():
            shutil.copy(video_path, output_path)
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
            shutil.copy(video_path, output_path)

    @staticmethod
    def _get_font_path() -> str:
        """获取当前平台可用的中文字体路径"""
        import sys
        import os

        candidates = []
        if sys.platform == "win32":
            candidates = [
                r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
                r"C:\Windows\Fonts\msyhbd.ttc",    # 微软雅黑粗体
                r"C:\Windows\Fonts\simhei.ttf",    # 黑体
                r"C:\Windows\Fonts\simsun.ttc",    # 宋体
            ]
        elif sys.platform == "darwin":
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]

        for path in candidates:
            if os.path.exists(path):
                return path
        return "Arial"

    def _transcode_output(
        self,
        input_path: str,
        output_path: str,
        output_format: OutputFormat,
    ):
        """转为最终输出格式"""
        keep_original = output_format.width <= 0 or output_format.height <= 0
        if keep_original:
            # 保持原分辨率，直接重编码（确保格式兼容）
            cmd = [
                "ffmpeg",
                "-i", input_path,
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
        else:
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

    @staticmethod
    def _probe_video_size(video_path: str) -> tuple[int, int, float]:
        """探测视频分辨率和帧率"""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_streams", "-select_streams", "v:0",
                    video_path,
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    w = int(streams[0].get("width", 0))
                    h = int(streams[0].get("height", 0))
                    fps_str = streams[0].get("r_frame_rate", "30/1")
                    try:
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            fps = float(num) / float(den)
                        else:
                            fps = float(fps_str)
                    except Exception:
                        fps = 30.0
                    return w, h, fps
        except Exception:
            pass
        return 0, 0, 30.0
