"""行程事实自动重建 — skills/travel-copywriting.md §9

不要让创作者从零写行程背景。素材元数据里已经记录了「哪天去了哪、
拍了什么、最晚拍到几点」，这些事实骨架可以自动重建。

创作者只需回答三个问题（都无法自动获得）：
  1. 原计划是什么？和实际有什么出入？   → 意外与不完美（§3.3）
  2. 哪个瞬间最不舒服？                 → 身体感受（§3.4）
  3. 为什么去这一趟？                   → 情绪基调

用法：
    facts = derive_trip_facts(AssetStore("data/assets_db/xinjiang_177.sqlite3"))
    print(facts.to_prompt_block())
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tools.common.models import TripFacts

logger = logging.getLogger(__name__)


def _media_created(path: str) -> datetime | None:
    """读拍摄时间。优先 QuickTime/EXIF 的 creation_time，回退文件 mtime。

    注意：文件 mtime 在网盘下载后会被改写，只能作为兜底。
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", path],
            capture_output=True, text=True, timeout=15,
        ).stdout
        import json
        tag = (json.loads(out).get("format", {}).get("tags", {}) or {}).get("creation_time")
        if tag:
            return datetime.fromisoformat(tag.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except Exception:
        return None


def derive_trip_facts(asset_store, destination: str | None = None) -> TripFacts:
    """从素材库重建行程事实骨架"""
    assets = (asset_store.filter_by_destination(destination)
              if destination else list(asset_store.list_all()))
    if not assets:
        logger.warning("素材库为空，无法重建行程事实")
        return TripFacts()

    times: list[datetime] = []
    by_day: dict[str, list[str]] = defaultdict(list)
    day_counter: Counter = Counter()
    dest_counter: Counter = Counter()

    for a in assets:
        ts = _media_created(a.source_path)
        if ts is None:
            continue
        times.append(ts)
        day = ts.strftime("%m-%d")
        day_counter[day] += 1
        if a.destination:
            dest_counter[a.destination] += 1
        for sc in a.scenes:
            cat = getattr(sc, "scene_category", "") or ""
            if cat and cat != "other":
                by_day[day].append(cat)

    if not times:
        logger.warning("所有素材都读不到拍摄时间（网盘下载可能已剥离元数据）")
        return TripFacts(total_clips=len(assets))

    times.sort()
    # 每天最晚 / 最早的拍摄时刻 —— "十点了，天还亮着"就是从这里来的
    latest = max(times, key=lambda t: (t.hour, t.minute))
    earliest = min(times, key=lambda t: (t.hour, t.minute))

    daily: dict[str, list[str]] = {}
    for day in sorted(by_day):
        top = [c for c, _ in Counter(by_day[day]).most_common(4)]
        daily[day] = top

    return TripFacts(
        date_range=f"{times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}",
        days=len({t.date() for t in times}),
        locations=[d for d, _ in dest_counter.most_common(8)],
        daily_scene_types=daily,
        earliest_shot_time=f"{earliest:%H:%M}",
        latest_shot_time=f"{latest:%H:%M}",
        densest_location=(dest_counter.most_common(1)[0][0] if dest_counter else ""),
        total_clips=len(assets),
    )


# 需要创作者回答的三个问题 —— 自动重建覆盖不到的部分
BRIEF_QUESTIONS = [
    "这趟原计划是什么？和实际有什么出入？（走不通的路、没等到的天气、临时改的行程）",
    "哪个瞬间最不舒服？（冷、累、堵车、干等）",
    "为什么去这一趟？",
]


def prompt_for_context() -> str:
    """交互式收集 trip_context。前两问的答案几乎必然通过替换测试，
    因为不适和意外天然是私人的。"""
    print("\n素材里的事实已自动提取。还需要三件只有你知道的事：\n")
    answers = []
    for i, q in enumerate(BRIEF_QUESTIONS, 1):
        a = input(f"{i}. {q}\n   > ").strip()
        if a:
            answers.append(a)
    return "。".join(answers)
