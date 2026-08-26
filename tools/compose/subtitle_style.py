"""ASS 字幕样式引擎 — 预设 + 入场动效 + 逐字点亮

原实现把 ASS 只当纯文本容器用：白字 + 3px 硬黑边 + 底部居中 + 整句一次性
出现 + 零动画，等于 2010 年电视台字幕。而 libass（ffmpeg 内置，零额外依赖）
支持的能力一个都没用上。

这里用起来的标签：
    \\fad(in,out)              淡入淡出
    \\t(t1,t2,\\fscx..\\fscy..)  时间变换（弹入缩放）
    \\k / \\kf                  逐字点亮（卡拉OK）
    \\blur \\bord \\shad         柔化粗描边，替代硬黑边
    \\an / \\pos                位置控制
    \\c \\3c                    主色 / 描边色

双轨设计：
    旁白轨 narration —— 跟 TTS 字级时间戳逐字点亮，底部小字
    金句轨 caption   —— copywriting 的错位字幕，中部大字带入场动效
两者文本不同（金句是 3-10 字错位句，旁白是完整叙述），
所以逐字同步只能挂在旁白轨上。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BASE_W, BASE_H = 1080, 1920


def rgb(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB → ASS 的 &HAABBGGRR（注意 ASS 是 BGR 顺序，且 alpha 反向）"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


@dataclass
class Preset:
    """一套字幕视觉预设"""
    name: str
    font_size: int                 # 基于 1080x1920 的字号
    primary: str = "#FFFFFF"       # 主色
    outline: str = "#000000"       # 描边色
    highlight: str = "#FFD700"     # 关键词 / 已唱过的字
    bold: bool = True
    border_style: int = 1          # 1=描边+阴影  3=不透明底块
    outline_w: int = 6
    shadow: int = 0
    blur: float = 2.0              # 柔化描边，硬边是土感的主要来源
    alignment: int = 2             # 2=底部居中 5=正中 8=顶部居中
    margin_v: int = 220
    spacing: int = 2
    entrance: str = "pop"          # pop | rise | fade | none

    def entrance_tags(self) -> str:
        """入场动效（内联覆盖标签）"""
        if self.entrance == "pop":
            # 从 62% 弹到 100%，160ms —— 短视频最常见的"蹦出来"
            return r"{\fad(110,90)\fscx62\fscy62\t(0,160,\fscx100\fscy100)}"
        if self.entrance == "rise":
            # 上移淡入：靠 \frz 不行，用 fad + 轻微缩放模拟，避免 \move 与对齐冲突
            return r"{\fad(160,110)\fscy88\t(0,180,\fscy100)}"
        if self.entrance == "fade":
            return r"{\fad(180,140)}"
        return ""


PRESETS: dict[str, Preset] = {
    # 抖音主流：大字重、粗柔描边、弹入、关键词变色
    "bold_pop": Preset(
        name="bold_pop", font_size=84, outline_w=7, blur=2.4,
        alignment=2, margin_v=240, entrance="pop",
    ),
    # 干净克制：半透明底块，适合风景片不抢画面
    "clean_minimal": Preset(
        name="clean_minimal", font_size=64, border_style=3,
        outline_w=14, blur=0.0, shadow=0, alignment=2,
        margin_v=200, entrance="fade", highlight="#7FD4FF",
    ),
    # 逐字点亮：配合 TTS 字级时间戳
    "karaoke": Preset(
        name="karaoke", font_size=72, outline_w=6, blur=2.0,
        alignment=2, margin_v=180, entrance="fade", highlight="#FFD700",
    ),
    # 旁白轨默认：小一号、压底、不抢金句
    "narration": Preset(
        name="narration", font_size=52, outline_w=5, blur=2.0,
        alignment=2, margin_v=110, entrance="fade", highlight="#FFD700",
    ),
}


@dataclass
class SubEvent:
    """一条待渲染的字幕"""
    start: float
    end: float
    text: str
    style: str = "Caption"
    # 字级时间戳 [(字, 起, 止)]，有则渲染成逐字点亮
    word_timings: list[tuple[str, float, float]] = field(default_factory=list)


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs, s = 0, s + 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_HL = re.compile(r"\*([^*]+)\*")


def _apply_highlight(text: str, preset: Preset) -> str:
    """把 *词* 渲染成高亮色 + 轻微放大"""
    def sub(m):
        w = m.group(1)
        c = rgb(preset.highlight)
        return rf"{{\c{c}\fscx112\fscy112}}{w}{{\c{rgb(preset.primary)}\fscx100\fscy100}}"
    return _HL.sub(sub, text)


def _karaoke_text(ev: SubEvent, preset: Preset) -> str:
    r"""用字级时间戳生成 \k 逐字点亮

    \k 的单位是厘秒。未唱到的字用 SecondaryColour，唱过的变 PrimaryColour——
    这是 ASS 卡拉OK的原生行为，不需要额外逻辑。
    """
    parts = []
    cursor = ev.start
    for ch, st, en in ev.word_timings:
        # 字与字之间的空档补成前一个字的延长，避免累计错位
        gap = max(0.0, st - cursor)
        if gap > 0.02:
            parts.append(rf"{{\k{int(round(gap * 100))}}}")
        dur_cs = max(1, int(round((en - st) * 100)))
        parts.append(rf"{{\k{dur_cs}}}{ch}")
        cursor = en
    return "".join(parts)


def build_ass(
    events: list[SubEvent],
    font_name: str = "Microsoft YaHei",
    play_w: int = BASE_W,
    play_h: int = BASE_H,
    caption_preset: str = "bold_pop",
    narration_preset: str = "narration",
) -> str:
    """生成完整 ASS 文件内容"""
    scale = play_h / BASE_H
    styles_used = {
        "Caption": PRESETS.get(caption_preset, PRESETS["bold_pop"]),
        "Narration": PRESETS.get(narration_preset, PRESETS["narration"]),
    }

    style_lines = []
    for sname, p in styles_used.items():
        style_lines.append(
            f"Style: {sname},{font_name},{int(p.font_size * scale)},"
            f"{rgb(p.primary)},{rgb(p.highlight)},{rgb(p.outline)},{rgb('#000000', 0x80)},"
            f"{-1 if p.bold else 0},0,0,0,100,100,{p.spacing},0,"
            f"{p.border_style},{max(1, int(p.outline_w * scale))},{int(p.shadow * scale)},"
            f"{p.alignment},60,60,{int(p.margin_v * scale)},1"
        )

    dialogues = []
    for ev in sorted(events, key=lambda e: e.start):
        if not ev.text.strip() and not ev.word_timings:
            continue
        p = styles_used.get(ev.style, styles_used["Caption"])

        blur = rf"\blur{p.blur}" if p.blur > 0 and p.border_style == 1 else ""
        if ev.word_timings:
            body = _karaoke_text(ev, p)
            lead = rf"{{\fad(120,100){blur}}}" if blur else r"{\fad(120,100)}"
        else:
            body = _apply_highlight(ev.text, p)
            lead = p.entrance_tags()
            if blur:
                lead = lead[:-1] + blur + "}" if lead else rf"{{{blur}}}"

        dialogues.append(
            f"Dialogue: 0,{_ass_time(ev.start)},{_ass_time(ev.end)},{ev.style},,0,0,0,,{lead}{body}"
        )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(style_lines) + """

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header + "\n".join(dialogues) + "\n"


def events_from_timeline(
    timeline: list[Any],
    narration_timings: dict[int, list[tuple[str, float, float]]] | None = None,
) -> list[SubEvent]:
    """把 EditDecision.timeline 转成双轨字幕事件

    Args:
        narration_timings: {item.order: [(字, 相对起, 相对止)]}
            来自 TTS 的字级时间戳；有则旁白轨逐字点亮，无则整句淡入
    """
    narration_timings = narration_timings or {}
    events: list[SubEvent] = []
    t = 0.0

    for item in timeline:
        dur = float(getattr(item, "duration_sec", 0.0))

        # 金句轨：大字动效
        cap = (getattr(item, "subtitle", "") or "").strip()
        if cap:
            events.append(SubEvent(start=t, end=t + dur, text=cap, style="Caption"))

        # 旁白轨：跟 TTS 逐字点亮
        nar = (getattr(item, "narration_text", "") or "").strip()
        if nar:
            wt = narration_timings.get(getattr(item, "order", -1)) or []
            events.append(SubEvent(
                start=t, end=t + dur, text=nar, style="Narration",
                # 时间戳是相对该镜头的，平移到全局
                word_timings=[(c, t + a, t + b) for c, a, b in wt],
            ))
        t += dur
    return events
