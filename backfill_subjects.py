#!/usr/bin/env python3
"""从已有 summary 回填结构化字段 —— 不重传视频，纯文本调用

背景：5f5cfd3 把打标层改成 VLM 结构化输出，但旧库里 177 个素材是老版本
入库的，subjects / time_of_day / mood 都是空的。全量重跑 VLM 要重新上传
177 个视频（¥6-20 + 几小时），而 summary 里其实已经含有物件信息：
  "夜幕下库尔德宁篝火晚会开场，人群围聚舞台前，火光烟雾"
  → subjects: ["篝火","舞台","围聚的人群","火光烟雾"]
  → time_of_day: night

用纯文本模型抽这些字段，成本降两个数量级（几毛钱、几分钟）。

拿不到的字段（summary 里通常不写，诚实留空）：
  shot_scale / camera_motion / usable_start_sec / usable_end_sec
  这些要景别和运镜判断，只能靠重跑 VLM。但它们主要服务 pacing 约束和
  检索，对文案影响不大 —— 文案要的是 subjects。

用法：
  python backfill_subjects.py --db data/assets_db/xinjiang_177.sqlite3 --dry-run
  python backfill_subjects.py --db data/assets_db/xinjiang_177.sqlite3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.common.asset_store import AssetStore
from tools.common.vocab import MOOD, TIME_OF_DAY, normalize_enum

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("backfill")

BATCH = 15

PROMPT = """从旅行视频的场景描述里抽取结构化字段。

## subjects 是这项任务的核心：画面里真实存在的具体物件

**不要填类别同义词或形容词**，只填能在画面里指认出来的东西。

判据：这个词能不能让人在画面里指出来？
  "湛蓝" 指不出来 ✗    "浮冰" 指得出来 ✓
  "辽阔" 指不出来 ✗    "散开的羊群" 指得出来 ✓

示例：
  描述："夜幕下库尔德宁篝火晚会开场，人群围聚舞台前，火光烟雾"
  → subjects: ["篝火", "舞台", "围聚的人群", "火光烟雾"]
     time_of_day: "night"   mood: "lively"

  描述："航拍俯瞰翠绿草原，牛羊散落其间，远山如黛，宁静辽阔"
  → subjects: ["散落的牛羊", "草原", "远山"]
     time_of_day: "unknown"  mood: "calm"
  （"翠绿""如黛""辽阔"是形容词，不是物件，不要放进 subjects）

## time_of_day 可选值
sunrise 日出 / golden_hour 金色时刻 / midday 正午 / overcast 阴天 /
blue_hour 蓝调时刻 / night 夜景 / indoor 室内 / unknown 描述里看不出

## mood 可选值
epic / calm / lively / curious / melancholic / cheerful / neutral

## 输入
下面是若干条场景描述，每条带一个 id。

## 输出
只返回 JSON 数组，不要 markdown 代码块：
[{"id":0,"subjects":["篝火","舞台"],"time_of_day":"night","mood":"lively"}]

规则：
- subjects 2-6 个，宁少勿滥。描述太笼统抽不出具体物件就给空数组
- 描述里没有依据的字段填 unknown / neutral，**不要猜**
"""


def call_llm(items: list[tuple[int, str]], model: str, api_key: str) -> dict[int, dict]:
    import dashscope
    from dashscope import Generation

    dashscope.api_key = api_key
    body = "\n".join(f"{i}. {s}" for i, s in items)

    try:
        resp = Generation.call(
            model=model,
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": body}],
            result_format="message",
            temperature=0.2,
        )
        if resp.status_code != 200:
            log.error(f"调用失败: {resp.code} - {resp.message}")
            return {}
        text = resp.output.choices[0].message.content
        if isinstance(text, list):
            text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        a, b = text.find("["), text.rfind("]") + 1
        if a < 0 or b <= a:
            log.error(f"无法解析: {text[:150]}")
            return {}
        return {int(r["id"]): r for r in json.loads(text[a:b]) if "id" in r}
    except Exception as e:
        log.error(f"异常: {e}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default="qwen-max")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写库")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个素材（试跑用）")
    ap.add_argument("--force", action="store_true", help="连已有 subjects 的也重抽")
    args = ap.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        sys.exit("✗ 需要 DASHSCOPE_API_KEY")

    store = AssetStore(args.db)
    assets = list(store.list_all())
    if args.limit:
        assets = assets[:args.limit]
    log.info(f"素材 {len(assets)} 个")

    # 收集待处理场景
    todo: list[tuple[int, str]] = []
    ref: list[tuple] = []          # (asset, scene)
    for a in assets:
        for sc in a.scenes:
            if not sc.summary.strip():
                continue
            if sc.subjects and not args.force:
                continue
            todo.append((len(ref), sc.summary))
            ref.append((a, sc))

    if not todo:
        print("没有待回填的场景（都已有 subjects，或用 --force 强制重抽）")
        return
    log.info(f"待回填场景 {len(todo)} 条，分 {(len(todo)+BATCH-1)//BATCH} 批")

    filled = 0
    empty = 0
    touched: set = set()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        got = call_llm(batch, args.model, api_key)
        for idx, _ in batch:
            r = got.get(idx)
            if not r:
                continue
            asset, sc = ref[idx]
            subs = [str(x).strip() for x in (r.get("subjects") or []) if str(x).strip()][:6]
            sc.subjects = subs
            sc.time_of_day = normalize_enum(r.get("time_of_day", ""), TIME_OF_DAY, "unknown")
            sc.mood = normalize_enum(r.get("mood", ""), MOOD, "neutral")
            if subs:
                filled += 1
            else:
                empty += 1
            touched.add(asset.asset_id)
            if args.dry_run and filled + empty <= 12:
                print(f"\n  {sc.summary[:52]}")
                print(f"  → subjects: {subs}")
                print(f"     time_of_day={sc.time_of_day}  mood={sc.mood}")
        log.info(f"  进度 {min(i+BATCH, len(todo))}/{len(todo)}")

    print(f"\n抽到物件 {filled} 条，抽不出（描述太笼统）{empty} 条")

    if args.dry_run:
        print("\n--dry-run 未写库。确认效果后去掉该参数重跑。")
        return

    by_id = {a.asset_id: a for a, _ in ref}
    for aid in touched:
        store.save(by_id[aid])
    print(f"已写回 {len(touched)} 个素材")
    print("\n下一步：跑 copywriting 看文案有没有变好")
    print("  python -m tools.edit.copywriter -e data/output/edit_decision.json \\")
    print('        -d 新疆 -c "你的真实行程背景"')


if __name__ == "__main__":
    main()
