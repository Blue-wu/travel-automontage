"""节拍网格吸附 — 让剪辑点落在 BPM 网格上

思路（方案 A）：不依赖具体某首歌，把所有切点吸附到 60/bpm 的网格。
这样「用哪首歌」和「怎么剪」就解耦了 —— 不需要曲库、不需要下载音频、
不需要 beat 检测，一次吸附就能配上该 BPM（及其整除/整倍 BPM）的任意歌曲。

⚠️ 一个网格只严格兼容成整除倍数关系的 BPM：
    120 网格 ⇒ 120 / 60 / 240 / 40 / 30 精确对齐
    120 网格 配 128 BPM ⇒ 每拍漂 0.031s，16 拍后差满一拍
所以不要指望"一次吸附通配 100-140"。正确用法是：
    抖音 App 里选好歌 → 查到它的 BPM → 用该 BPM 重新吸附 → 再渲染
重新吸附是纯计算，不重跑检索/剧本/文案决策，几毫秒的事。

关键实现点：**按累计时间轴吸附，不是逐镜头独立四舍五入。**
逐个独立取整会让误差累积漂移，第 20 个镜头可能已经偏了半拍。
这里始终以「吸附后的累计时刻」为下一镜起点，误差不传递。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tools.common.models import EditDecision

logger = logging.getLogger(__name__)


@dataclass
class SnapRow:
    order: int
    before: float
    after: float
    beats: float

    @property
    def delta(self) -> float:
        return round(self.after - self.before, 3)


@dataclass
class SnapReport:
    bpm: int
    subdivision: int
    grid: float
    rows: list[SnapRow] = field(default_factory=list)
    total_before: float = 0.0
    total_after: float = 0.0
    max_drift: float = 0.0          # 吸附前后累计时刻的最大偏差
    clamped: int = 0                # 因超出延长上限而回退的镜头数

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.bpm,
            "subdivision": self.subdivision,
            "grid_sec": round(self.grid, 4),
            "total_before": round(self.total_before, 2),
            "total_after": round(self.total_after, 2),
            "max_drift": round(self.max_drift, 3),
            "clamped": self.clamped,
            "compatible_bpms": compatible_bpms(self.bpm),
        }


def compatible_bpms(bpm: int, lo: int = 60, hi: int = 200) -> list[int]:
    """严格兼容的 BPM —— 与 bpm 成整除/整倍关系的才精确对齐

    诚实地只列真正对得上的，不做"接近就算"的模糊承诺。
    """
    out = set()
    for k in (1, 2, 3, 4):
        if bpm * k <= hi:
            out.add(bpm * k)
        if bpm % k == 0 and bpm // k >= lo:
            out.add(bpm // k)
    return sorted(out)


def snap_to_grid(
    edit_decision: EditDecision,
    bpm: int = 120,
    subdivision: int = 1,
    min_beats: float = 1.0,
    allow_extend_beats: float = 1.0,
) -> SnapReport:
    """把时间线的切点吸附到节拍网格（原地修改 edit_decision）

    Args:
        bpm: 目标速度
        subdivision: 网格细分。1=整拍（默认，切点落在拍上，卡点感最强）
            2=半拍（允许切在反拍，快剪段更灵活但卡点感弱）
        min_beats: 单镜最短占几格，防止吸附成 0
        allow_extend_beats: 允许比原时长延长的上限（格）。
            延长意味着要多用素材 —— 因为 usable_end_sec 通常小于 end_sec
            （打标时已裁掉落幅），所以 out_sec 之后一般还有余量，
            小幅延长是安全的；超过上限则向下取整而不是硬拉长。

    Returns:
        SnapReport，含逐镜对照和累计漂移
    """
    grid = 60.0 / float(bpm) / max(1, subdivision)
    rep = SnapReport(bpm=bpm, subdivision=subdivision, grid=grid)

    if not edit_decision.timeline:
        return rep

    t_snap = 0.0        # 吸附后的累计时刻
    t_ideal = 0.0       # 原始累计时刻，用于算漂移

    for item in edit_decision.timeline:
        orig = float(item.duration_sec)
        t_ideal += orig

        # ★ 锚定【理想累计时刻】吸附，而不是 t_snap + orig。
        # 后者看似"按累计"，但 t_snap 恒在网格上，
        # round((t_snap+orig)/grid) == t_snap/grid + round(orig/grid)，
        # 数学上等价于逐镜独立取整，误差照样随机游走累积。
        # 锚定 t_ideal 才能把漂移恒定压在半格以内。
        end = round(t_ideal / grid) * grid

        # 下限：至少 min_beats 格
        if end - t_snap < min_beats * grid - 1e-9:
            end = t_snap + min_beats * grid

        # 上限：延长不得超过 allow_extend_beats 格，超了就向下取一格
        if end - t_snap > orig + allow_extend_beats * grid + 1e-9:
            cand = end - grid
            if cand - t_snap >= min_beats * grid - 1e-9:
                end = cand
                rep.clamped += 1

        new_dur = round(end - t_snap, 4)
        item.duration_sec = new_dur
        item.out_sec = round(item.in_sec + new_dur, 4)

        rep.rows.append(SnapRow(
            order=item.order, before=round(orig, 3),
            after=new_dur, beats=round(new_dur / grid, 2),
        ))
        t_snap = end
        rep.max_drift = max(rep.max_drift, abs(t_snap - t_ideal))

    rep.total_before = round(t_ideal, 3)
    rep.total_after = round(t_snap, 3)
    edit_decision.total_duration_sec = rep.total_after

    logger.info(
        f"节拍吸附完成: bpm={bpm} 网格={grid:.3f}s "
        f"{rep.total_before:.1f}s → {rep.total_after:.1f}s "
        f"最大漂移 {rep.max_drift:.3f}s"
        + (f"，{rep.clamped} 个镜头因延长超限而向下取格" if rep.clamped else "")
    )
    return rep


class BeatGridSnapper:
    """供 pipeline stage 调用的包装

    必须排在 copywriting 之前 —— 吸附会改变镜头时长，
    而字幕字数上限是按时长算的（skills/travel-copywriting.md §5）。
    """

    def snap(
        self,
        edit_decision: EditDecision | dict[str, Any],
        bpm: int = 120,
        subdivision: int = 1,
    ) -> EditDecision:
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)
        snap_to_grid(edit_decision, bpm=bpm, subdivision=subdivision)
        return edit_decision


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="把剪辑点吸附到 BPM 网格")
    ap.add_argument("-e", "--edit-decision", required=True)
    ap.add_argument("-b", "--bpm", type=int, default=120)
    ap.add_argument("--subdivision", type=int, default=1,
                    help="1=整拍（默认，卡点最实）2=半拍（快剪段更灵活）")
    ap.add_argument("-o", "--out", default="", help="写回路径，默认只预览")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ed = EditDecision(**json.loads(Path(args.edit_decision).read_text(encoding="utf-8")))
    rep = snap_to_grid(ed, bpm=args.bpm, subdivision=args.subdivision)

    print(f"\n网格 {rep.grid:.3f}s  ({args.bpm} BPM / {args.subdivision} 细分)")
    print(f"{'#':>3}  {'原时长':>7}  {'吸附后':>7}  {'变化':>7}  拍数")
    print("-" * 46)
    for r in rep.rows:
        print(f"{r.order:>3}  {r.before:>6.2f}s  {r.after:>6.2f}s  "
              f"{r.delta:>+6.2f}s  {r.beats:>4.1f}")
    print("-" * 46)
    print(f"总时长 {rep.total_before:.2f}s → {rep.total_after:.2f}s")
    print(f"最大累计漂移 {rep.max_drift:.3f}s（逐镜独立取整会漂到几百毫秒）")
    if rep.clamped:
        print(f"{rep.clamped} 个镜头因延长超限而向下取格")
    print(f"精确兼容的 BPM: {compatible_bpms(args.bpm)}")
    print("配其他 BPM 的歌 → 用那首歌的 BPM 重跑本命令即可，不必重跑流水线")

    if args.out:
        Path(args.out).write_text(
            json.dumps(ed.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    _main()
