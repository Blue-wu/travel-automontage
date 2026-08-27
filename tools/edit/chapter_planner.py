"""章节规划 — 把一整天的素材切成有叙事顺序的章节

解决的问题：
  原来 _estimate_output_duration 把成片硬夹在 15-60 秒
  （estimated = max(15.0, min(60.0, estimated))），
  于是一整天 30 分钟素材被压进 40 多秒 —— 每个地点只分到 2-3 秒，
  蜻蜓点水。而且镜头是「按画质排序后随机取」，把拍摄时间顺序打乱了。

  结果是既没故事也没连贯。但这不是缺了什么，是把本来有的东西弄丢了：
  **连贯性的来源是现成的 —— 拍摄时间。**
  早上出发 → 路上 → 抵达 → 玩 → 傍晚回，这本身就是叙事，不用编。

做法：
  按拍摄时间的自然间隔聚类成章节（吃饭/赶路/休息 = 天然断点），
  再按体裁分配每章预算。
    vlog  模式：章节顺序串成一条长片，章节间给标题卡
    split 模式：每章独立出一条短视频（一天素材发一条是浪费）

时间从哪来：优先解析文件名里的时间戳（DJI/VID/IMG/PXL 等命名都带），
拿不到再退 ffprobe creation_time，最后退文件 mtime（网盘下载会改写，最不可信）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 常见相机/手机命名里的时间戳
_TS_PATTERNS = [
    re.compile(r"(\d{8})(\d{6})"),            # DJI_20260527141956_
    re.compile(r"(\d{8})[_\-](\d{6})"),       # VID_20260527_141956 / PXL_...
]

SCENE_CN = {
    "snow_mountain": "雪山", "grassland": "草原", "lake": "湖泊", "river": "河谷",
    "forest": "森林", "canyon": "峡谷", "desert": "沙漠", "sea": "海边",
    "road": "在路上", "aerial": "航拍", "sunset": "日落", "starry_sky": "星空",
    "sky": "天空", "architecture": "老街", "city": "城里", "flower": "花海",
    "reflection": "倒影", "people": "人物", "food": "吃的", "animal": "动物",
    "other": "其他",
}


def parse_shot_time(path: str) -> datetime | None:
    """从文件名解析拍摄时间；失败则退 ffprobe / mtime"""
    base = os.path.basename(path)
    for pat in _TS_PATTERNS:
        m = pat.search(base)
        if not m:
            continue
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            continue

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", path],
            capture_output=True, text=True, timeout=10,
        ).stdout
        tag = (json.loads(out).get("format", {}).get("tags", {}) or {}).get("creation_time")
        if tag:
            return datetime.fromisoformat(tag.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass

    try:
        # 网盘下载会改写 mtime，最不可信，仅兜底
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


@dataclass
class ChapterItem:
    shot_at: datetime
    source_path: str
    scenes: list[dict[str, Any]]

    @property
    def duration(self) -> float:
        return sum(float(s.get("end_sec", 0)) - float(s.get("start_sec", 0))
                   for s in self.scenes)


@dataclass
class Chapter:
    index: int
    items: list[ChapterItem]
    title: str = ""
    budget_sec: float = 0.0

    @property
    def start(self) -> datetime:
        return self.items[0].shot_at

    @property
    def end(self) -> datetime:
        return self.items[-1].shot_at

    @property
    def n_scenes(self) -> int:
        return sum(len(i.scenes) for i in self.items)

    @property
    def available_sec(self) -> float:
        return sum(i.duration for i in self.items)

    def top_categories(self, n: int = 3) -> list[tuple[str, int]]:
        from collections import Counter
        c = Counter(s.get("scene_category", "other")
                    for i in self.items for s in i.scenes)
        return c.most_common(n)

    def distinctive_subjects(self, global_df: dict[str, int], n: int = 2) -> list[str]:
        """本章的特色物件：本章频次高、但全局不普遍的 subject

        纯按主导 scene_category 起标题会撞车（实测 4 个章节都叫「草原」），
        因为 66% 素材被误分类成 people，回退逻辑总是落到同一个次主导类别。
        改用类 TF-IDF 的做法，取真正区分得开的具体物件。
        """
        from collections import Counter
        local = Counter(x for i in self.items for s in i.scenes
                        for x in (s.get("subjects") or []))
        if not local:
            return []
        scored = []
        for w, cnt in local.items():
            if not w or w.endswith("镜头"):     # 占位污染词，见 backfill 的已知问题
                continue
            df = global_df.get(w, 1)
            scored.append((cnt / (df ** 0.7), w))     # 局部频次 / 全局普遍度
        scored.sort(reverse=True)
        return [w for _, w in scored[:n]]

    def auto_title(self, global_df: dict[str, int] | None = None) -> str:
        """时间 + 特色物件 —— 章节标题天然就是好字幕，
        「14:17 羊羔」比「人间值得」有信息量得多"""
        if global_df:
            subs = self.distinctive_subjects(global_df, 1)
            if subs:
                return f"{self.start:%H:%M} {subs[0]}"
        cats = self.top_categories(2)
        pick = next((c for c, _ in cats if c not in ("people", "other")), None)
        name = SCENE_CN.get(pick or (cats[0][0] if cats else "other"), "")
        return f"{self.start:%H:%M} {name}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title or self.auto_title(),
            "start": self.start.strftime("%Y-%m-%d %H:%M"),
            "end": self.end.strftime("%H:%M"),
            "n_assets": len(self.items),
            "n_scenes": self.n_scenes,
            "available_sec": round(self.available_sec, 1),
            "budget_sec": round(self.budget_sec, 1),
            "top_categories": self.top_categories(),
            "source_paths": [i.source_path for i in self.items],
        }


def _split_by_gaps(items: list[ChapterItem], k: int) -> list[list[ChapterItem]]:
    """按最大的 k-1 个时间间隔切分"""
    if k <= 1 or len(items) <= 1:
        return [items]
    gaps = sorted(
        ((items[i].shot_at - items[i - 1].shot_at).total_seconds(), i)
        for i in range(1, len(items))
    )
    cuts = sorted(i for _, i in gaps[-(k - 1):])
    out, prev = [], 0
    for cut in cuts + [len(items)]:
        if cut > prev:
            out.append(items[prev:cut])
        prev = cut
    return out


def plan_chapters(
    items: list[ChapterItem],
    target: int = 6,
    min_assets: int = 4,
    max_assets: int | None = None,
) -> list[Chapter]:
    """按拍摄时间聚类成章节

    纯 gap 切分会留下两种坏结果，这里都处理掉：
      - 孤立单段章节（实测会切出 1 段/0.5 分钟的章）→ 合并到时间上最近的邻居
      - 过大章节（实测有 100 段的章）→ 用其内部最大 gap 继续拆

    max_assets=None 时按 target 自适应：短体裁（target 小）不强拆，
    否则 50 秒会被拆成 8 章、每章 5.8 秒 —— 那还是蜻蜓点水，
    和「一天压进 40 秒」是同一个毛病。
    """
    items = sorted(items, key=lambda x: x.shot_at)
    if not items:
        return []

    if max_assets is None:
        # 允许每章装到平均量的 2.5 倍，章节数才不会被强拆逻辑顶穿
        max_assets = max(12, int(len(items) / max(1, target) * 2.5))

    groups = _split_by_gaps(items, target)

    # 拆分过大章节
    changed = True
    while changed:
        changed = False
        out = []
        for g in groups:
            if len(g) > max_assets and len(g) > 1:
                parts = _split_by_gaps(g, 2)
                if len(parts) > 1:
                    out.extend(parts)
                    changed = True
                    continue
            out.append(g)
        groups = out

    # 合并过小章节到相邻（选时间间隔更近的一侧）
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i, g in enumerate(groups):
            if len(g) >= min_assets:
                continue
            if i == 0:
                tgt = 1
            elif i == len(groups) - 1:
                tgt = i - 1
            else:
                d_prev = (g[0].shot_at - groups[i - 1][-1].shot_at).total_seconds()
                d_next = (groups[i + 1][0].shot_at - g[-1].shot_at).total_seconds()
                tgt = i - 1 if d_prev <= d_next else i + 1
            groups[tgt] = sorted(groups[tgt] + g, key=lambda x: x.shot_at)
            groups.pop(i)
            merged = True
            break

    chapters = [Chapter(index=i + 1, items=g) for i, g in enumerate(groups)]

    from collections import Counter
    global_df = Counter(x for it in items for s in it.scenes
                        for x in set(s.get("subjects") or []))
    used: set[str] = set()
    for ch in chapters:
        cands = ch.distinctive_subjects(global_df, 4) or [""]
        pick = next((c for c in cands if c and c not in used), cands[0])
        if pick:
            used.add(pick)
            ch.title = f"{ch.start:%H:%M} {pick}"
        else:
            ch.title = ch.auto_title()
    logger.info(
        f"章节规划: {len(items)} 段素材 → {len(chapters)} 章 "
        f"({', '.join(str(len(c.items)) for c in chapters)} 段)"
    )
    return chapters


def allocate_budget(
    chapters: list[Chapter],
    total_sec: float,
    min_chapter_sec: float = 8.0,
) -> None:
    """按各章可用素材量按比例分配成片时长（原地写入 budget_sec）

    不再硬夹 15-60 秒 —— 时长由体裁决定，不由一行 clamp 决定。
    """
    if not chapters:
        return
    total_avail = sum(c.available_sec for c in chapters) or 1.0
    for c in chapters:
        c.budget_sec = max(min_chapter_sec, total_sec * c.available_sec / total_avail)
    # 归一回目标总时长
    s = sum(c.budget_sec for c in chapters)
    if s > 0:
        for c in chapters:
            c.budget_sec = round(c.budget_sec * total_sec / s, 1)


# ── 体裁预设 ──────────────────────────────────────────────
FORMATS = {
    # 15-20s：靠冲击力和密度，不需要故事
    "shorts": {"total_sec": 18, "avg_shot": 1.6, "chapters": 1},
    # 40-60s：最难的体裁 —— 太长撑不住纯节奏，太短讲不完故事
    "single": {"total_sec": 50, "avg_shot": 2.6, "chapters": 3},
    # 3-8min：靠时间线叙事，连贯性由结构给
    "vlog": {"total_sec": 300, "avg_shot": 3.2, "chapters": 6},
}


def load_items_from_store(asset_store, destination: str | None = None) -> list[ChapterItem]:
    """从素材库读出带拍摄时间的条目"""
    assets = (asset_store.filter_by_destination(destination)
              if destination else list(asset_store.list_all()))
    out, no_time = [], 0
    for a in assets:
        ts = parse_shot_time(a.source_path)
        if ts is None:
            no_time += 1
            continue
        out.append(ChapterItem(
            shot_at=ts,
            source_path=a.source_path,
            scenes=[s.model_dump() if hasattr(s, "model_dump") else s for s in a.scenes],
        ))
    if no_time:
        logger.warning(f"{no_time} 个素材解析不到拍摄时间，已跳过")
    return sorted(out, key=lambda x: x.shot_at)


def _main() -> None:
    import argparse
    import sys as _sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="按拍摄时间把素材切成章节")
    ap.add_argument("--db", required=True, help="素材库路径")
    ap.add_argument("-f", "--format", default="vlog", choices=list(FORMATS))
    ap.add_argument("-d", "--destination", default=None)
    ap.add_argument("-t", "--target", type=int, default=0, help="章节数，默认按体裁")
    ap.add_argument("--total-sec", type=float, default=0, help="总时长，默认按体裁")
    ap.add_argument("-o", "--out", default="", help="导出章节计划 JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.common.asset_store import AssetStore

    fmt = FORMATS[args.format]
    items = load_items_from_store(AssetStore(args.db), args.destination)
    if not items:
        raise SystemExit("✗ 素材库为空，或全部解析不到拍摄时间")

    chapters = plan_chapters(items, target=args.target or fmt["chapters"])
    allocate_budget(chapters, args.total_sec or fmt["total_sec"])

    span_h = (items[-1].shot_at - items[0].shot_at).total_seconds() / 3600
    avail = sum(i.duration for i in items) / 60
    print(f"\n素材 {len(items)} 段  拍摄跨度 {span_h:.1f} 小时  可用 {avail:.1f} 分钟")
    print(f"体裁 {args.format}  目标 {args.total_sec or fmt['total_sec']:.0f}s\n")
    print(f"{'#':>2}  {'章节':<18} {'时间':<14} {'素材':>5} {'可用':>7} {'出片':>7}")
    print("-" * 62)
    for ch in chapters:
        print(f"{ch.index:>2}  {ch.title:<18} {ch.start:%H:%M}–{ch.end:%H:%M}   "
              f"{len(ch.items):>4} {ch.available_sec/60:>6.1f}分 {ch.budget_sec:>6.1f}s")
    print("-" * 62)
    print(f"合计 {sum(c.budget_sec for c in chapters):.0f}s，{len(chapters)} 章")

    print("\n一拆多（每章单独出一条短视频）：")
    for ch in chapters:
        print(f"  片{ch.index}「{ch.title}」 {len(ch.items)} 段 "
              f"→ 可出 {min(60, max(12, int(ch.available_sec * 0.2)))}s")

    if args.out:
        Path(args.out).write_text(
            json.dumps([c.to_dict() for c in chapters], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n已导出 {args.out}")


if __name__ == "__main__":
    _main()
