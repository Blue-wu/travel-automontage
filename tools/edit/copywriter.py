"""文案生成器 — copywriting stage

与 script_generator 的根本区别：
- script_generator 决定「结构」（哪个镜头放在哪），是确定性的
- copywriter  决定「说什么」，由 LLM 在**看到完整时间线之后一次性**生成

为什么必须在时间线定稿后、且一次生成整条：
- 字幕要卡镜头时长（3 秒的镜头塞不下 12 个字）
- 首尾要呼应、时序不能矛盾 —— 逐镜头独立生成结构上就产生不了呼应

规范见 skills/travel-copywriting.md，该文件由 pipeline_runner 通过
stage 的 `skill:` 字段注入 skill_text 参数。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tools.common.config import get_settings
from tools.common.models import CreativeBrief, EditDecision, TripFacts

logger = logging.getLogger(__name__)


# 兜底禁用清单（正常情况下从 skill_text 的「禁用清单」小节解析，此处仅作 fallback）
FALLBACK_BANNED = [
    "人间值得", "治愈瞬间", "这就是远方", "来了就懂了",
    "什么都不想了", "时间慢下来了", "被治愈的一天", "安静且自由",
    "风知道答案", "和自己和解了", "路上才是答案", "世界比想象中大",
    "所有奔波都值了", "整个人都麻了", "一眼万年", "梦里的样子",
    "名不虚传", "亲眼见到才懂", "比想象中震撼", "还会再来的",
    "故事还很长", "带着风景离开", "说走就走", "出发永远不晚",
    "绝美", "震撼", "治愈", "惊艳", "宝藏", "神仙", "天花板", "秒杀", "无敌",
]


def max_chars(duration_sec: float) -> int:
    """字幕字数上限 —— 字幕停留时间 = 镜头时长，读不完等于没有

    见 skills/travel-copywriting.md §4
    """
    if duration_sec < 1.5:
        return 0          # 快剪段不上字幕
    if duration_sec < 2.5:
        return 6
    if duration_sec < 4.0:
        return 10
    return 14


class Copywriter:
    """LLM 驱动的字幕文案生成器

    用法：
        cw = Copywriter()
        ed = cw.write(edit_decision, brief=brief, skill_text=open(...).read())
    """

    def __init__(self, model: str = "", api_key: str = ""):
        s = get_settings()
        self.api_key = api_key or s.dashscope.api_key
        # 纯文本任务，用文本模型即可，比视觉模型便宜得多
        self.model = model or "qwen-max"

    # ── 对外主入口 ────────────────────────────────────────────

    def write(
        self,
        edit_decision: EditDecision | dict[str, Any],
        brief: CreativeBrief | dict[str, Any] | None = None,
        trip_facts: TripFacts | dict[str, Any] | None = None,
        skill_text: str = "",
    ) -> EditDecision:
        """为整条时间线生成字幕，写回 edit_decision.timeline[*].subtitle

        返回同类型对象，因此 compose 阶段无需任何改动。
        """
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)
        if isinstance(brief, dict):
            brief = CreativeBrief(**brief)
        if isinstance(trip_facts, dict):
            trip_facts = TripFacts(**trip_facts)
        brief = brief or CreativeBrief()

        if not edit_decision.timeline:
            logger.warning("时间线为空，跳过文案生成")
            return edit_decision

        if not self.api_key:
            logger.warning(
                "无 DASHSCOPE_API_KEY —— 保留 script_generator 的模板字幕。"
                "注意：模板字幕是 random.choice 抽签，无故事性，见 "
                "skills/travel-copywriting.md"
            )
            return edit_decision

        banned = self._parse_banned(skill_text) or FALLBACK_BANNED
        banned = list(dict.fromkeys(banned + list(brief.avoid)))

        shots = self._build_shot_table(edit_decision)
        logger.info(f"文案生成: {len(shots)} 个镜头，一次性生成整条时间线")

        result = self._call_llm(shots, brief, trip_facts, skill_text, banned)
        if result is None:
            logger.warning("文案生成失败，保留原字幕")
            return edit_decision

        # 校验 → 不合格则带着问题清单让模型改一轮
        problems = self._validate(result, shots, banned)
        if problems:
            logger.info(f"首轮不合格（{len(problems)} 项），触发修订")
            repaired = self._call_llm(
                shots, brief, trip_facts, skill_text, banned,
                problems=problems, previous=result,
            )
            if repaired is not None:
                result = repaired
                problems = self._validate(result, shots, banned)

        if problems:
            logger.warning(f"修订后仍有 {len(problems)} 项问题，将强制截断处理")

        self._apply(edit_decision, result, shots)
        return edit_decision

    # ── 内部 ──────────────────────────────────────────────────

    def _build_shot_table(self, ed: EditDecision) -> list[dict[str, Any]]:
        """把时间线转成给模型看的镜头表"""
        shots = []
        t = 0.0
        for item in ed.timeline:
            d = item.duration_sec
            shots.append({
                "order": item.order,
                "at": round(t, 1),
                "duration": round(d, 1),
                "max_chars": max_chars(d),
                # 画面描述 + 具体物件 —— 错位手法的抓手（skill §7）
                # "湖泊"抓不出手法，"没化完的浮冰"才能抓出"冰敷"
                "visual": item.visual_summary or item.narration_text or "",
                "subjects": item.visual_tags or [],
                "source": item.source_path,
            })
            t += d
        return shots

    def _build_prompt(
        self,
        shots: list[dict],
        brief: CreativeBrief,
        trip_facts: TripFacts | None,
        skill_text: str,
        banned: list[str],
        problems: list[str] | None,
        previous: dict | None,
    ) -> tuple[str, str]:
        system = skill_text.strip() or "你是旅行 Vlog 字幕撰稿人。"
        system += (
            "\n\n---\n\n"
            "你现在要为一条已定稿的时间线撰写全部画面字幕。核心要求：\n"
            "1. 【预测测试】遮住字幕只看画面，观众能猜到的一律重写。"
            "描述画面的、抒情的，都是猜得到的。\n"
            "2. 每条有字的字幕标注所用手法 device（A 场景误读 / B 宏大降格 / "
            "C 身份错位 / D 数值荒诞 / E 错位归因），平实句留空字符串。"
            "标不出手法的通常就是套话。\n"
            "3. 【控制用力】整条片子错位手法只用 2-4 处，其余有字的镜头写平实句。"
            "全是梗，梗就不响了。\n"
            "4. 严格遵守每个镜头的 max_chars；max_chars=0 必须留空。\n"
            "5. 覆盖率 40%-60%，约一半镜头留空。\n"
            "6. 必须有一组首尾呼应；同片内不复用同一手法。\n"
            "7. 写不出错位就写平实事实，**绝不退回抒情套话**。\n"
            "只输出 JSON，不要 markdown 代码块。"
        )

        parts = [f"## 目的地\n{brief.destination}\n"]
        if brief.persona:
            parts.append(f"## 口吻\n{brief.persona}\n")
        if brief.trip_context:
            parts.append(f"## 真实行程背景（活人感的核心原料，必须用上）\n{brief.trip_context}\n")
        else:
            parts.append(
                "## ⚠ 未提供行程背景\n"
                "缺少真实经历，请只依据画面描述和行程事实写，"
                "**宁可留空也不要编造事实或写套话**。\n"
            )
        if trip_facts:
            block = trip_facts.to_prompt_block()
            if block:
                parts.append(f"## 行程事实（自素材元数据自动重建，可直接引用）\n{block}\n")

        parts.append(f"## 禁用词\n{'、'.join(banned)}\n")

        parts.append("## 时间线\n")
        parts.append("| # | 起始 | 时长 | 字数上限 | 画面 | 画面物件（错位抓手）|")
        parts.append("|---|---|---|---|---|---|")
        for sh in shots:
            cap = "**必须留空**" if sh["max_chars"] == 0 else str(sh["max_chars"])
            subj = "、".join(sh.get("subjects") or []) or "—"
            parts.append(
                f"| {sh['order']} | {sh['at']}s | {sh['duration']}s | {cap} "
                f"| {sh['visual'] or '（无描述）'} | {subj} |"
            )

        if problems and previous:
            parts.append("\n## 上一版的问题，请逐条修正\n")
            for p in problems:
                parts.append(f"- {p}")
            parts.append(f"\n上一版输出：\n{json.dumps(previous, ensure_ascii=False)}")

        parts.append(
            "\n## 输出格式\n"
            '{"subtitles":['
            '{"order":1,"text":"独库还在放假","device":"C","reason":"身份错位·埋呼应"},'
            '{"order":2,"text":"","device":"","reason":"快剪留空"},'
            '{"order":6,"text":"湖还在冰敷","device":"A","reason":"抓手=浮冰，结冰→冰敷"}],'
            '"callback_pair":[1,15],'
            '"self_check":{"prediction_test_passed":true,"banned_words_used":[],'
            '"coverage_rate":0.47,"device_count":4,"devices_used":["A","C","E"],'
            '"strongest_line":"太阳在新疆加班"}}'
        )
        return system, "\n".join(parts)

    def _call_llm(
        self,
        shots: list[dict],
        brief: CreativeBrief,
        trip_facts: TripFacts | None,
        skill_text: str,
        banned: list[str],
        problems: list[str] | None = None,
        previous: dict | None = None,
    ) -> dict | None:
        system, user = self._build_prompt(
            shots, brief, trip_facts, skill_text, banned, problems, previous
        )
        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = self.api_key
            resp = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                result_format="message",
                temperature=0.8,      # 文案要多样性，但不能失控
                top_p=0.9,
            )
            if resp.status_code != 200:
                logger.error(f"文案模型调用失败: {resp.code} - {resp.message}")
                return None
            text = resp.output.choices[0].message.content
            if isinstance(text, list):
                text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
            return self._parse_json(text)
        except Exception as e:
            logger.error(f"文案生成异常: {e}")
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            a, b = text.find("{"), text.rfind("}") + 1
            if a >= 0 and b > a:
                try:
                    return json.loads(text[a:b])
                except json.JSONDecodeError:
                    pass
        logger.error(f"文案 JSON 解析失败: {text[:200]}")
        return None

    @staticmethod
    def _parse_banned(skill_text: str) -> list[str]:
        """从 skill 的「禁用清单」小节解析，保持单一事实来源。

        按标题匹配而非章节号 —— 规范改版重排章节时不会失效。
        """
        if not skill_text:
            return []
        m = re.search(r"##\s*\d+\.\s*禁用清单[^\n]*\n(.*?)(?=\n##\s)", skill_text, re.S)
        if not m:
            return []
        section = m.group(1)
        words: list[str] = []
        # 先取围栏块，并把它们从正文剔除，否则残留的 ``` 会干扰行内反引号配对
        for block in re.findall(r"```(.*?)```", section, re.S):
            words += re.split(r"[\s、,，/]+", block.strip())
        section = re.sub(r"```.*?```", " ", section, flags=re.S)
        for inline in re.findall(r"`([^`]+)`", section):
            words += re.split(r"[\s、,，/]+", inline)
        return [w for w in {w.strip() for w in words} if len(w) >= 2]

    def _validate(self, result: dict, shots: list[dict], banned: list[str]) -> list[str]:
        """返回问题清单，空列表表示合格"""
        problems: list[str] = []
        by_order = {s["order"]: s for s in shots}
        subs = {s.get("order"): (s.get("text") or "").strip()
                for s in result.get("subtitles", [])}

        filled = 0
        for order, shot in by_order.items():
            text = subs.get(order, "")
            cap = shot["max_chars"]
            if text:
                filled += 1
            if cap == 0 and text:
                problems.append(
                    f"第 {order} 镜时长仅 {shot['duration']}s，必须留空，但写了「{text}」"
                )
            elif cap and len(text) > cap:
                problems.append(
                    f"第 {order} 镜超字数（上限 {cap}，实际 {len(text)}）：「{text}」"
                )
            for b in banned:
                if b and b in text:
                    problems.append(f"第 {order} 镜命中禁用词「{b}」：「{text}」")
                    break

        rate = filled / max(1, len(by_order))
        if not (0.35 <= rate <= 0.65):
            problems.append(
                f"字幕覆盖率 {rate:.0%} 超出 40%-60% 区间（当前 {filled}/{len(by_order)} 条有字）"
            )

        # 错位手法：数量和多样性（skill §2 / §3）
        devices = [
            (s.get("device") or "").strip().upper()
            for s in result.get("subtitles", [])
            if (s.get("text") or "").strip()
        ]
        used = [d for d in devices if d in {"A", "B", "C", "D", "E"}]
        if len(used) < 2:
            problems.append(
                f"错位手法只用了 {len(used)} 处，至少 2 处（skill §2）。"
                "全是平实句会平淡"
            )
        elif len(used) > 4:
            problems.append(
                f"错位手法用了 {len(used)} 处，上限 4 处（skill §3）。"
                "全是梗，梗就不响了 —— 改几条为平实句"
            )
        dup = {d for d in used if used.count(d) > 1}
        if dup:
            problems.append(f"手法 {'、'.join(sorted(dup))} 重复使用，同片内应轮换（skill §6.3）")

        if not result.get("callback_pair"):
            problems.append("缺少首尾呼应（skill §6.2）")
        return problems

    def _apply(self, ed: EditDecision, result: dict, shots: list[dict]) -> None:
        """写回时间线；仍超字数的做硬截断兜底"""
        subs = {s.get("order"): (s.get("text") or "").strip()
                for s in result.get("subtitles", [])}
        by_order = {s["order"]: s for s in shots}

        for item in ed.timeline:
            text = subs.get(item.order, "")
            cap = by_order.get(item.order, {}).get("max_chars", 10)
            if cap == 0:
                text = ""
            elif len(text) > cap:
                text = text[:cap]
            item.subtitle = text

        n = sum(1 for i in ed.timeline if i.subtitle)
        logger.info(
            f"文案完成: {n}/{len(ed.timeline)} 条字幕 "
            f"(覆盖率 {n / max(1, len(ed.timeline)):.0%})"
        )


# ── 独立命令行入口 ────────────────────────────────────────────
# 不跑整条流水线，直接给已有的 edit_decision.json 配字幕：
#   python -m tools.edit.copywriter -e data/output/edit_decision.json \
#          -d 新疆 -c "五月底自驾伊犁，独库还没开，绕了果子沟"

def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="为已有时间线生成画面字幕")
    ap.add_argument("-e", "--edit-decision", required=True, help="edit_decision.json 路径")
    ap.add_argument("-d", "--destination", default="", help="目的地")
    ap.add_argument("-c", "--trip-context", default="",
                    help="真实行程背景 1-3 句（手法 B/D 的唯一原料，强烈建议填）")
    ap.add_argument("-p", "--persona", default="克制、不煽情、像跟朋友讲事")
    ap.add_argument("-s", "--skill", default="skills/travel-copywriting.md")
    ap.add_argument("-o", "--out", default="", help="写回路径，默认只打印不落盘")
    ap.add_argument("--avoid", nargs="*", default=[], help="额外禁用词")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from pathlib import Path

    ed_path = Path(args.edit_decision)
    if not ed_path.exists():
        raise SystemExit(f"✗ 找不到 {ed_path}")
    ed = EditDecision(**json.loads(ed_path.read_text(encoding="utf-8")))

    skill_path = Path(args.skill)
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    if not skill_text:
        print(f"⚠ 规范文件不存在: {skill_path} —— 将只用兜底禁用清单，效果会明显变差")
    if not args.trip_context:
        print("⚠ 未提供 --trip-context —— 手法 B/D 无原料，只能写平实句")

    brief = CreativeBrief(
        destination=args.destination,
        trip_context=args.trip_context,
        persona=args.persona,
        avoid=list(args.avoid),
    )

    result = Copywriter().write(ed, brief=brief, skill_text=skill_text)

    print(f"\n{'#':>3}  {'时长':>6}  {'上限':>4}  字幕")
    print("-" * 60)
    filled = 0
    for it in result.timeline:
        cap = max_chars(it.duration_sec)
        text = it.subtitle or "—"
        if it.subtitle:
            filled += 1
        print(f"{it.order:>3}  {it.duration_sec:>5.1f}s  {cap:>4}  {text}")
    total = len(result.timeline)
    print("-" * 60)
    print(f"覆盖率 {filled}/{total} = {filled / max(1, total):.0%}"
          f"（目标 40%-60%）")

    if args.out:
        Path(args.out).write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    _main()
