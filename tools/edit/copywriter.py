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


# 兜底禁用清单（正常情况下从 skill_text 的 §2 代码块解析，此处仅作 fallback）
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
                # 画面描述 —— 质量直接决定文案质量（skill §6）
                "visual": item.narration_text or "",
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
            "你现在要为一条已定稿的时间线撰写全部画面字幕。"
            "严格遵守上面的规范，特别是：\n"
            "1. 每一条都必须通过【替换测试】——把地名换掉后句子还成立的，一律重写\n"
            "2. 禁用清单里的词及其近义变体一律不得出现\n"
            "3. 严格遵守每个镜头给出的 max_chars；max_chars=0 的镜头必须留空\n"
            "4. 整条覆盖率 40%-60%，即约一半镜头应当留空\n"
            "5. 必须有一组首尾呼应、一条意外、一条身体感受\n"
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
        parts.append("| # | 起始 | 时长 | 字数上限 | 画面 |")
        parts.append("|---|---|---|---|---|")
        for s in shots:
            cap = "**必须留空**" if s["max_chars"] == 0 else str(s["max_chars"])
            parts.append(
                f"| {s['order']} | {s['at']}s | {s['duration']}s | {cap} | {s['visual'] or '（无描述）'} |"
            )

        if problems and previous:
            parts.append("\n## 上一版的问题，请逐条修正\n")
            for p in problems:
                parts.append(f"- {p}")
            parts.append(f"\n上一版输出：\n{json.dumps(previous, ensure_ascii=False)}")

        parts.append(
            "\n## 输出格式\n"
            '{"subtitles":[{"order":1,"text":"独库还没开","reason":"意外·埋呼应"}],'
            '"callback_pair":[1,15],'
            '"self_check":{"substitution_test_passed":true,"banned_words_used":[],'
            '"coverage_rate":0.47,"has_accident":"","has_body_feeling":"","screenshot_line":""}}'
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
        """从 skill 的 §2 代码块解析禁用清单，保持单一事实来源"""
        if not skill_text:
            return []
        m = re.search(r"##\s*2\.[^\n]*\n(.*?)(?=\n##\s)", skill_text, re.S)
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

        chk = result.get("self_check") or {}
        if not chk.get("has_accident"):
            problems.append("缺少「意外与不完美」类字幕（skill §3.3）")
        if not chk.get("has_body_feeling"):
            problems.append("缺少「身体感受」类字幕（skill §3.4）")
        if not result.get("callback_pair"):
            problems.append("缺少首尾呼应（skill §5.2）")
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
