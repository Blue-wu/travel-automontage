"""章节 → 时间线 —— 把章节规划变成可渲染的 EditDecision

为什么单独一层而不是改 script_generator：
  原来的选片是「按画质排序后从前 5 个随机取一个」，
  把拍摄时间顺序彻底打乱，所以既没故事也没连贯。
  这里绕开那条路径，直接从章节构建时间线，保证：
    - 章内按拍摄时间排序（叙事顺序 = 现成的，不用编）
    - 章间按时间先后串联
    - 每章开头挂章节标题（「23:07 篝火晚会」本身就是好字幕）

选片算法：时间分桶取最优
  把一章的场景按时间均分成 N 个桶，每桶取画质最高的一个。
  这样既保住了时间顺序，又不会连着用相邻的重复镜头，
  比「全局按画质排序」更适合叙事。
"""

from __future__ import annotations

import logging
from typing import Any

import re

from tools.common.models import EditDecision, OutputFormat, TimelineItem

logger = logging.getLogger(__name__)

# 「{类别中文名}镜头」= VLM 打标失败、走 CLIP 回退产生的占位描述。
# 全库 93/270 (34%) 是这种，画面内容其实是未知的。
# 旧的 _enhance_quality 按关键词命中数把它们的 quality 抬到 0.68+，
# 所以「取画质最高」反而优先选中它们 —— 必须显式降权。
_PLACEHOLDER = re.compile(
    r"^(人物人像|公路自驾|草原田野|野生动物|森林树木|天空云海|雪山山脉|湖泊水景"
    r"|河流瀑布|大海海滩|沙漠戈壁|峡谷地貌|日落日出|星空银河|花海花朵|建筑人文"
    r"|城市风光|美食小吃|航拍俯视|倒影镜面|其他)镜头$"
)


def is_placeholder(scene: dict) -> bool:
    return bool(_PLACEHOLDER.match((scene.get("summary") or "").strip()))


def usable_span(scene: dict) -> float:
    a = float(scene.get("usable_start_sec") or scene.get("start_sec", 0.0))
    b = float(scene.get("usable_end_sec") or scene.get("end_sec", a))
    return max(0.0, b - a)


def pick_score(scene: dict) -> float:
    """选片打分：画质为主，占位描述重罚"""
    q = float(scene.get("quality", 0.0))
    return q * (0.35 if is_placeholder(scene) else 1.0)


def _pick_by_time_buckets(
    scenes: list[dict], n: int, min_quality: float, min_shot: float,
) -> list[dict]:
    """按时间均分成 n 桶，每桶取得分最高的一个（保持时间序）

    可用区间短于 min_shot 的直接排除 —— 否则会被 min(per, avail) 砍成
    0.3 秒的镜头，闪一下就没了，等于废帧。
    """
    usable = [s for s in scenes
              if float(s.get("quality", 0)) >= min_quality
              and usable_span(s) >= min_shot]
    if not usable:
        usable = [s for s in scenes if usable_span(s) >= min_shot]
    if not usable:
        usable = sorted(scenes, key=lambda x: -usable_span(x))[:n]
    usable = sorted(usable, key=lambda s: (s.get("_shot_at"), s.get("start_sec", 0)))

    if n >= len(usable):
        return usable

    picked = []
    step = len(usable) / n
    for i in range(n):
        lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
        bucket = usable[lo:hi]
        if bucket:
            picked.append(max(bucket, key=pick_score))
    return picked


def _avoid_repeat_category(picked: list[dict]) -> list[dict]:
    """相邻同 scene_category 时，与后面最近的异类交换一次

    数据里 camera_motion 53% 为空、shot_scale 只有三档，不足以支撑
    完整的景别嵌套/动静交替约束，这里只做最低限度的去重复。
    """
    out = list(picked)
    for i in range(1, len(out) - 1):
        if out[i].get("scene_category") != out[i - 1].get("scene_category"):
            continue
        for j in range(i + 1, len(out)):
            if out[j].get("scene_category") != out[i - 1].get("scene_category"):
                out[i], out[j] = out[j], out[i]
                break
    return out


def build_timeline(
    chapters: list[Any],
    avg_shot: float = 3.2,
    min_shot: float = 1.8,
    max_shot: float = 6.0,
    min_quality: float = 0.45,
    title_cards: bool = True,
    output_format: dict[str, Any] | None = None,
) -> EditDecision:
    """把章节规划构建成 EditDecision

    Args:
        chapters: chapter_planner.plan_chapters 的输出（需先 allocate_budget）
        title_cards: 每章第一个镜头挂上章节标题作为字幕
    """
    timeline: list[TimelineItem] = []
    total = 0.0

    for ch in chapters:
        # 摊平该章所有场景，并记住各自来自哪个素材、拍摄时间
        scenes: list[dict] = []
        for it in ch.items:
            for s in it.scenes:
                d = dict(s)
                d["_source_path"] = it.source_path
                d["_shot_at"] = it.shot_at
                scenes.append(d)
        if not scenes:
            continue

        budget = float(getattr(ch, "budget_sec", 0.0)) or ch.available_sec * 0.2
        n = max(1, min(len(scenes), round(budget / avg_shot)))
        picked = _avoid_repeat_category(
            _pick_by_time_buckets(scenes, n, min_quality, min_shot))
        if not picked:
            continue

        per = max(min_shot, min(max_shot, budget / len(picked)))
        # 标题挂位：章节标题是「HH:MM 特色物件」，物件来自全章统计，
        # 而首镜只是时间序第一个，两者本就不一致 —— 直接挂首镜会文不对题
        # （实测「11:45 连绵远山」挂到了"红色灯柱、人群穿行"上）。
        # 优先挂到真正包含该物件的镜头，其次挂第一个非占位镜头。
        title = getattr(ch, "title", "") or ""
        kw = title.split(" ", 1)[1] if " " in title else ""
        title_at = None
        if kw:
            for i, x in enumerate(picked):
                if kw in (x.get("subjects") or []) or kw in (x.get("summary") or ""):
                    title_at = i
                    break
        if title_at is None:
            title_at = next(
                (i for i, x in enumerate(picked) if not is_placeholder(x)), 0)

        for idx, s in enumerate(picked):
            # 用可用区间避开起幅落幅
            a = float(s.get("usable_start_sec") or s.get("start_sec", 0.0))
            b = float(s.get("usable_end_sec") or s.get("end_sec", a + per))
            avail = max(min_shot, b - a)
            dur = round(min(per, avail), 3)

            subtitle = ""
            if title_cards and idx == title_at:
                subtitle = getattr(ch, "title", "") or ""

            timeline.append(TimelineItem(
                order=len(timeline) + 1,
                asset_id=str(s.get("asset_id", "")),
                source_path=s["_source_path"],
                in_sec=round(a, 3),
                out_sec=round(a + dur, 3),
                duration_sec=dur,
                transition_in="fade" if idx == 0 else "cut",
                transition_duration=0.4 if idx == 0 else 0.0,
                subtitle=subtitle,
                # 合并后 composer 读的是 top_subtitle（双时间轴的视觉层），
                # subtitle 仅作旧字段兼容 —— 两个都写，避免章节标题渲染不出来
                top_subtitle=subtitle,
                narration_text="",
                visual_summary=s.get("summary", "") or "",
                visual_tags=list(s.get("subjects") or s.get("visual_tags") or []),
            ))
            total += dur

    ed = EditDecision(
        timeline=timeline,
        output_format=OutputFormat(**(output_format or {})),
        total_duration_sec=round(total, 2),
    )
    logger.info(
        f"章节时间线: {len(chapters)} 章 → {len(timeline)} 个镜头，"
        f"总时长 {total:.1f}s"
    )
    return ed


class ChapterTimelineBuilder:
    """供 pipeline stage 调用"""

    def build(
        self,
        chapters: list[Any],
        avg_shot: float = 3.2,
        title_cards: bool = True,
        output_format: dict[str, Any] | None = None,
    ) -> EditDecision:
        return build_timeline(
            chapters, avg_shot=avg_shot, title_cards=title_cards,
            output_format=output_format,
        )


def _main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="章节 → 可渲染时间线")
    ap.add_argument("--db", required=True)
    ap.add_argument("-f", "--format", default="vlog")
    ap.add_argument("-t", "--target", type=int, default=0)
    ap.add_argument("--total-sec", type=float, default=0)
    ap.add_argument("-o", "--out", default="", help="导出 edit_decision.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.common.asset_store import AssetStore
    from tools.edit.chapter_planner import (
        FORMATS, allocate_budget, load_items_from_store, plan_chapters,
    )

    fmt = FORMATS[args.format]
    items = load_items_from_store(AssetStore(args.db))
    chs = plan_chapters(items, target=args.target or fmt["chapters"])
    allocate_budget(chs, args.total_sec or fmt["total_sec"])
    ed = build_timeline(chs, avg_shot=fmt["avg_shot"])

    print(f"\n{'#':>3} {'时刻':>7} {'时长':>6}  章节/画面")
    print("-" * 74)
    t = 0.0
    ch_starts = {}
    acc = 0
    for ch in chs:
        ch_starts[acc + 1] = ch.title
        b = float(ch.budget_sec)
        n = max(1, round(b / fmt["avg_shot"]))
        acc += n
    for it in ed.timeline:
        mark = f"  ★ {it.subtitle}" if it.subtitle else ""
        print(f"{it.order:>3} {t:>6.1f}s {it.duration_sec:>5.1f}s  "
              f"{(it.visual_summary or '')[:34]}{mark}")
        t += it.duration_sec
    print("-" * 74)
    print(f"{len(ed.timeline)} 个镜头，总时长 {ed.total_duration_sec:.1f}s "
          f"（{ed.total_duration_sec/60:.1f} 分钟）")

    if args.out:
        Path(args.out).write_text(
            json.dumps(ed.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"已导出 {args.out}")


if __name__ == "__main__":
    _main()
