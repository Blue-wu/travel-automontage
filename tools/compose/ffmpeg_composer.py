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

from tools.common.models import EditDecision, OutputFormat, TimelineItem, VoiceoverConfig

logger = logging.getLogger(__name__)


class FFmpegComposer:
    """FFmpeg 视频合成器"""

    def compose(
        self,
        edit_decision: EditDecision | dict,
        output_path: str = "data/output/final_video.mp4",
        *,
        voiceover: dict | None = None,
    ) -> str:
        """从剪辑决策渲染最终视频

        Args:
            edit_decision: 剪辑决策（EditDecision 对象或 dict）
            output_path: 输出文件路径
            voiceover: 可选，覆盖 edit_decision.voiceover 的配音配置
                       （用于 copywriter 之后 voiceover_regeneration 产出的新配音，
                        含 segments 时长元数据，用于字幕-配音同步对齐）

        Returns:
            输出文件路径
        """
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)

        # 如果传入了新的 voiceover（regenerate 产物，含 segments 时长），覆盖旧的
        if voiceover is not None:
            try:
                if isinstance(voiceover, dict):
                    edit_decision.voiceover = VoiceoverConfig(**voiceover)
                elif isinstance(voiceover, VoiceoverConfig):
                    edit_decision.voiceover = voiceover
            except Exception as e:
                logger.warning(
                    f"传入的 voiceover 无法解析为 VoiceoverConfig，继续使用原配置: {e}"
                )

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

            # Step 3+: 准备配音元数据（用于字幕 Bottom 时间轴）
            # 新方案：配音原速播放（_add_voiceover 不再 atempo 拉伸），
            # 所以字幕 Bottom 时间轴 = 配音段原声时长累加，无需计算 atempo_ratio。
            vo_dict: dict | None = None
            if edit_decision.voiceover and edit_decision.voiceover.path:
                vo_dict = (
                    edit_decision.voiceover.model_dump()
                    if hasattr(edit_decision.voiceover, "model_dump")
                    else dict(edit_decision.voiceover)
                )
                # 日志：打印配音与画面时长对比，方便人工核对
                try:
                    probe_v = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_format", str(concat_path)],
                        capture_output=True, text=True, check=True,
                    )
                    video_duration_sec = float(
                        json.loads(probe_v.stdout).get("format", {}).get("duration", 0)
                    )
                    vo_path = vo_dict.get("path", "")
                    if vo_path and Path(vo_path).exists():
                        probe_vo = subprocess.run(
                            ["ffprobe", "-v", "quiet", "-print_format", "json",
                             "-show_format", vo_path],
                            capture_output=True, text=True, check=True,
                        )
                        vo_duration_sec = float(
                            json.loads(probe_vo.stdout).get("format", {}).get("duration", 0)
                        )
                        logger.info(
                            f"[字幕-配音同步] 视频总长={video_duration_sec:.2f}s "
                            f"配音原声={vo_duration_sec:.2f}s (原速播放，不拉伸)"
                        )
                except Exception as e:
                    logger.debug(f"probe 配音/视频时长失败（不影响渲染）: {e}")

            # Step 3: 添加字幕（检测 top/bottom 任一有内容都加）
            subtitled_path = concat_path
            has_sub = any(
                (item.subtitle or item.top_subtitle or item.bottom_subtitle)
                for item in edit_decision.timeline
            )
            if has_sub:
                subtitled_path = tmpdir / "subtitled.mp4"
                self._add_subtitles(
                    str(concat_path),
                    edit_decision.timeline,
                    str(subtitled_path),
                    edit_decision.output_format,
                    voiceover=vo_dict,
                    # atempo_ratio 不再传：字幕直接按配音段原声时长累计
                )

            # Step 4: 添加旁白配音（如果有）
            voiced_path = subtitled_path
            if vo_dict and vo_dict.get("path"):
                voiced_path = tmpdir / "voiced.mp4"
                self._add_voiceover(
                    str(subtitled_path),
                    vo_dict,
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
        *,
        voiceover: dict | None = None,
        atempo_ratio: float | None = None,
    ):
        """添加字幕（硬字幕烧录） — 使用 ASS 字幕文件
        双字幕系统（双时间轴，单一真源对齐）：
        1) TopStyle    左上角横排金句 → 时间轴=画面镜头累计时长（视觉层，跟画面切镜走）
        2) BottomStyle 底部配音跟读字幕 → 时间轴=【配音段原声时长】累计（音频节奏层，跟配音走）
           注：atempo_ratio 参数已废弃，仅为向后兼容保留，强制按 1.0 处理（配音原速）。
        """
        has_any = any(item.top_subtitle or item.bottom_subtitle for item in timeline)
        if not has_any:
            shutil.copy(input_path, output_path)
            return

        font_path = self._get_font_path()
        ass_content = self._generate_ass_subtitles(
            timeline,
            font_path,
            output_format,
            voiceover=voiceover,
            atempo_ratio=atempo_ratio,
        )

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
        *,
        voiceover: dict | None = None,
        atempo_ratio: float | None = None,
    ) -> str:
        """生成 ASS 格式字幕 — 双字幕系统 + 双时间轴（单一真源）：
        1) TopStyle:   左上角横排错位金句（白底黑字小盒子）
                        → 时间轴 = 画面镜头时长累加（视觉层，跟画面切镜走）
        2) BottomStyle: 底部配音跟读字幕（黑字白描边大字）
                        → 时间轴 = 【配音段原声时长】累加（音频节奏层，跟配音走，原速）

        单一真源定义（解决三种风格配音-字幕不同步的核心）：
        · Bottom 字幕必须和 TTS 配音的实际朗读节奏严格一致 ——
          配音在 _add_voiceover 中**原速播放**（不 atempo 拉伸），
          所以字幕也直接用配音段原声时长累加，二者天然对齐。
        · 每个有 narration_text 的 timeline item，对应 segments 中的一个 scene 段。
        · 先跳过 hook 段的时长（hook 是开场白，没有镜头字幕对应），
          再按 scene 段顺序累加 vo_t，
          Bottom 字幕窗口 = [vo_t, vo_t + seg_dur - 0.05s 气口]

        参数 atempo_ratio 仅为向后兼容保留，已不再使用（强制 ratio=1.0）。

        基准分辨率：横屏 1920x1080 基准，4K 下 scale=2
        """
        # ⚠️ 原来是 base_w, base_h = 1920, 1080（横屏基准），但输出是竖屏
        # 1080x1920，于是 scale = play_h / base_h = 1920/1080 = 1.78 ——
        # 拿竖屏的高去除横屏的高没有意义，字号被整体放大 1.78 倍，
        # 底部字幕 147px 在 1080 宽画面上一行只放得下 7 个字。
        # 字幕大小该按【画面宽度】衡量（可读性取决于占宽比），基准 1080 宽。
        base_w, base_h = 1080, 1920

        if output_format and output_format.width > 0 and output_format.height > 0:
            play_w, play_h = output_format.width, output_format.height
        else:
            play_w, play_h = base_w, base_h

        scale = play_w / base_w      # 按宽度缩放：1080→1.0，4K竖屏 2160→2.0

        # ═══════════════════════════════════════════════════════════
        # 准备 Bottom 字幕的「配音节奏真源」：
        #  - 从 voiceover.segments 里筛选出 scene 段（跳过 hook/cta）
        #  - 每段 Bottom 字幕窗口长度 = 该段原声 dur（不再 × ratio）
        #  - 开场先跳过 hook 段（hook 是配音开场白，镜头 item 没对它）
        #  - ratio 强制为 1.0：配音原速播放，字幕按原声时长累计
        # ═══════════════════════════════════════════════════════════
        all_segments: list[dict] = []
        scene_segments: list[dict] = []
        hook_total_raw = 0.0
        ratio = 1.0  # ★ 原速播放，不拉伸
        vo_ready = False

        if voiceover:
            segs = voiceover.get("segments")
            if isinstance(segs, list) and len(segs) > 0:
                all_segments = segs
                scene_segments = [s for s in all_segments if s.get("kind") == "scene"]
                hook_total_raw = sum(
                    float(s.get("duration_sec") or 0)
                    for s in all_segments if s.get("kind") == "hook"
                )
                vo_ready = True

        # 兜底策略：没有 segments 元数据时 → 退化成画面时长轴（保证字幕至少能显示）
        if not vo_ready:
            def _fallback_scene_durs() -> list[float]:
                return [
                    max(0.4, float(item.duration_sec))
                    for item in timeline if item.narration_text
                ]
            scene_segments = [
                {"kind": "scene", "duration_sec": d}
                for d in _fallback_scene_durs()
            ]
            hook_total_raw = 0.0

        # ═══════════════════════════════════════════════════════════
        # 初始化两条时间轴
        # ═══════════════════════════════════════════════════════════
        video_t = 0.0       # Top 字幕：画面镜头时长累计
        # Bottom 字幕：先加上 hook 段原声时长偏移（hook 说完，scene 第一段才开始）
        vo_t = hook_total_raw
        scene_idx = 0       # 指向下一个要消费的 scene_segments
        narr_items_count = sum(1 for it in timeline if it.narration_text)
        safe_scene_count = min(len(scene_segments), narr_items_count)

        events: list[str] = []

        for item in timeline:
            clip_dur = max(0.3, item.duration_sec)

            # ── Top 金句（视觉层）: 用画面镜头时间轴 ──
            top_s = video_t
            top_e = video_t + clip_dur - 0.2
            if top_e <= top_s:
                top_e = top_s + max(0.3, clip_dur - 0.05)
            if item.top_subtitle:
                events.append(
                    "Dialogue: 0,{ss},{es},TopStyle,,0,0,0,,{txt}".format(
                        ss=self._seconds_to_ass_time(top_s),
                        es=self._seconds_to_ass_time(top_e),
                        txt=item.top_subtitle,
                    )
                )

            # ── Bottom 跟读（音频节奏层）: 用配音段原声时长累计 ──
            if item.narration_text and scene_idx < safe_scene_count:
                raw_dur = float(scene_segments[scene_idx].get("duration_sec") or 0)
                if raw_dur <= 0:
                    raw_dur = max(0.4, len(item.narration_text) * 0.28)
                bot_win_dur = raw_dur  # ★ 原速，不乘 ratio
                bot_s = vo_t
                bot_e = vo_t + bot_win_dur - 0.05
                if bot_e <= bot_s:
                    bot_e = bot_s + max(0.3, bot_win_dur - 0.01)

                # 用 narration_text（镜头旁白）作为字幕文本；
                # 若该 item 显式指定了 bottom_subtitle，优先用那个
                bot_txt = item.bottom_subtitle or item.narration_text
                if bot_txt:
                    events.append(
                        "Dialogue: 0,{ss},{es},BottomStyle,,0,0,0,,{txt}".format(
                            ss=self._seconds_to_ass_time(bot_s),
                            es=self._seconds_to_ass_time(bot_e),
                            txt=bot_txt,
                        )
                    )
                vo_t += bot_win_dur
                scene_idx += 1
            elif item.bottom_subtitle:
                # 没对应 narration_text 但指定了 bottom_subtitle → 仍然按镜头窗口展示
                bot_s = video_t
                bot_e = video_t + clip_dur - 0.2
                if bot_e <= bot_s:
                    bot_e = bot_s + max(0.3, clip_dur - 0.05)
                events.append(
                    "Dialogue: 0,{ss},{es},BottomStyle,,0,0,0,,{txt}".format(
                        ss=self._seconds_to_ass_time(bot_s),
                        es=self._seconds_to_ass_time(bot_e),
                        txt=item.bottom_subtitle,
                    )
                )

            # 推进画面累计（Top 时间轴）
            video_t += clip_dur

        font_name = self._get_font_name(font_path)

        # 两种 style 的参数：
        # TopStyle: 左上角横排小盒子， Alignment=7（左上），白底圆角
        # BottomStyle: 底部横排大字， Alignment=2（中下），黑字白描边
        outline_top = max(2, int(3 * scale))
        margin_top_l = int(80 * scale)   # TopStyle 左边距
        margin_top_t = int(80 * scale)   # TopStyle 上边距
        outline_bot = max(3, int(4 * scale))
        margin_v_bot = int(180 * scale)  # BottomStyle 下边距

        # 顶部字号：1080p 基准 56px → 4K 约 120px（金句别太大）
        top_font_size = int(56 * scale)
        # 底部字号：1080p 基准 72px → 4K 约 165px，+15% 补偿
        bot_font_size = int(72 * scale)
        bot_font_size = int(bot_font_size * 1.15)

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TopStyle,{font_name},{top_font_size},&H00101010,&H000000FF,&H00FFFFFF,&H90FFFFFF,-1,0,0,0,100,100,0,0,3,{outline_top},0,7,{margin_top_l},0,{margin_top_t},1
Style: BottomStyle,{font_name},{bot_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline_bot},2,2,20,20,{margin_v_bot},1

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
        """备用：drawtext 方式添加字幕（同步修复：配音时间轴 + 横屏基准字号）"""
        drawtext_parts = []

        # 配音累计时间轴（同 _generate_ass_subtitles）
        chars_per_sec = 2.5
        voiceover_t = 0.0
        font_path = self._get_font_path()
        import os
        font_path_escaped = font_path.replace("\\", "/").replace(":", r"\:")

        # 横屏 1080p 基准； drawtext 是最终输出分辨率下的像素，4K下直接要 200px 左右
        base_font_size_1080p = 100

        for item in timeline:
            if item.subtitle:
                if item.narration_text:
                    dur = max(1.0, len(item.narration_text) / chars_per_sec)
                else:
                    dur = max(1.0, item.duration_sec * 0.6)
                start = voiceover_t
                end = voiceover_t + dur

                text = item.subtitle.replace(":", r"\:").replace("'", r"'\''").replace("%", r"\%")

                style = item.subtitle_style or {}
                # item 里是 1080p 基准 font_size，按画面 h/1080 缩放后再 × 1.15 渲染补偿
                raw_base = int(style.get("font_size", base_font_size_1080p))
                est_play_h = 2160  # 4K 预估（drawtext 没有 format 信息时）
                scale = est_play_h / 1080
                font_size = int(raw_base * scale * 1.15)
                position = style.get("position", "bottom")

                # 位置留更大的 margin（因为字大了）
                if position == "bottom":
                    y_expr = "h-text_h-200"
                elif position == "center":
                    y_expr = "(h-text_h)/2"
                else:
                    y_expr = "200"

                drawtext_parts.append(
                    f"drawtext=text='{text}':"
                    f"fontfile='{font_path_escaped}':"
                    f"fontsize={font_size}:fontcolor=white:"
                    f"borderw=4:bordercolor=black:"
                    f"x=(w-text_w)/2:y={y_expr}:"
                    f"enable='between(t,{start:.2f},{end:.2f})'"
                )

                # 累计配音时间
                if item.narration_text:
                    narr_end = len(item.narration_text) / chars_per_sec
                    gap = min(1.2, max(0.4, item.duration_sec - narr_end))
                    voiceover_t += narr_end + max(0.4, gap)
                else:
                    voiceover_t = end + 0.4
            else:
                if item.narration_text:
                    d = max(0.8, len(item.narration_text) / chars_per_sec)
                else:
                    d = 0.8
                voiceover_t += d

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
        """添加旁白配音（原速播放，不拉伸）

        配音对齐策略（与 _generate_ass_subtitles 严格一致）：
        1. 配音**原速播放**，不做 atempo 拉伸 ——
           字幕 Bottom 时间轴是基于「配音段原声时长累加」生成的，
           配音若被拉伸，实际朗读节奏会与字幕窗口错位。
        2. loudnorm 仅做响度归一（不改变时长），atrim 截断到视频长度避免越界。
        3. 原视频声音压低到 0.3，配音与原声 amix，duration=first 跟随视频结束。
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

        # 原视频声音压低到 0.3，配音原速 → loudnorm 归一化 → atrim 截断到视频长度
        # 不再使用 atempo 拉伸：保证字幕（按配音段原声时长累计）与配音实际节奏严格一致
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", vo_path,
            "-filter_complex",
            f"[0:a]volume=0.3[original];"
            f"[1:a]atrim=0:{duration:.2f},volume={volume},"
            f"loudnorm=I=-16:TP=-1.5:LRA=11[vo];"
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
            logger.info(
                f"旁白配音已添加（原速）: 配音 {vo_duration:.1f}s, 视频时长 {duration:.1f}s "
                f"{'[配音超长已截断]' if vo_duration > duration else '[配音短于画面，末段静音]'}"
            )
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
