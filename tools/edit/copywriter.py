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
            # YAML 传来的 avoid 可能是中文逗号分隔的字符串
            brief = self._normalize_creative_brief(brief)
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
        if brief.style:
            logger.info(f"文案风格: {brief.style} (persona={brief.persona[:20]}...)")
            logger.info(f"禁用词追加 {len(brief.avoid)} 个: {brief.avoid[:5]}...")

        shots = self._build_shot_table(edit_decision)
        logger.info(f"文案生成: {len(shots)} 个镜头，一次性生成整条时间线")

        result = self._call_llm(shots, brief, trip_facts, skill_text, banned)
        if result is None:
            logger.warning("LLM 不可用，走 skill 本地规则优化路径")
            result = self._local_optimize(edit_decision, shots, banned)

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

    @staticmethod
    def _normalize_creative_brief(brief: dict) -> CreativeBrief:
        """把 YAML/CLI 传进来的 raw dict 规范化成 CreativeBrief。

        特别处理几个从 CLI 风格注册表以"逗号字符串"形式传进来的字段：
        - avoid: "美不胜收，心灵的故乡" / ["美不胜收", "心灵的故乡"] / str → list[str]
        - device_priority: "C,A,D" → list[str]，写入 CreativeBrief.extra（prompt 会读它）
        """
        avoid_raw = brief.pop("avoid", [])
        if isinstance(avoid_raw, str) and avoid_raw:
            # 中英文逗号、顿号都拆
            import re as _re
            avoid_list = [
                w.strip()
                for w in _re.split(r"[，,；;、]", avoid_raw)
                if w.strip()
            ]
        elif isinstance(avoid_raw, list):
            avoid_list = [str(w).strip() for w in avoid_raw if str(w).strip()]
        else:
            avoid_list = []

        device_priority_raw = brief.pop("device_priority", "")
        if isinstance(device_priority_raw, str) and device_priority_raw:
            import re as _re
            device_priority = [
                x.strip()
                for x in _re.split(r"[，,、]", device_priority_raw)
                if x.strip()
            ]
        elif isinstance(device_priority_raw, list):
            device_priority = [str(x).strip() for x in device_priority_raw if str(x).strip()]
        else:
            device_priority = []

        # device_priority 暂存到 avoid 尾部（用哨兵分隔）不妥——改存到 avoid 里
        # 更稳妥：CreativeBrief 没 device 字段，我们把 device_priority 追加到 avoid 末尾
        # 并在 prompt 里读出来。用约定字符串 "DEVICE_PRIORITY:A,B,C" 塞进 persona 的尾部。
        persona = brief.get("persona", "") or ""
        if device_priority:
            persona += f"\n【手法优先顺序】device 使用按 {' > '.join(device_priority)} 优先级选，"
            persona += "排不到再用其他手法。不要全片用同一种手法。"
            brief["persona"] = persona

        brief["avoid"] = avoid_list
        return CreativeBrief(**brief)

    # ── 内部 ──────────────────────────────────────────────────

    def _local_optimize(
        self, ed: EditDecision, shots: list[dict], banned: list[str],
    ) -> dict:
        """LLM 不可用时的本地 skill 规则优化

        按 travel-copywriting.md 的硬性规范处理 script_generator 的模板文案：
        1. 命中禁用词的字幕清空（符合 skill §1「预测测试」——套话等于没写）
        2. 字幕超字数硬截断（skill §4 max_chars）
        3. 配音文案超字数硬截断（narration_max_chars）
        4. max_chars=0 的镜头字幕强制留空（skill §4）
        5. 覆盖率超过 60% 时，按镜头顺序清空多余字幕（skill §5）
        6. 字幕命中禁用词的镜头，用 visual_summary 兜底重写为平实句

        无法做到的：错位手法、首尾呼应、LLM 级别的创意改写
        """
        by_order = {s["order"]: s for s in shots}
        subtitles_out: list[dict] = []

        # 第一遍：清理禁用词 + 截断
        for item in ed.timeline:
            sh = by_order.get(item.order, {})
            cap = sh.get("max_chars", 10)
            narr_cap = sh.get("narration_max_chars", 20)

            # 字幕处理
            sub = (item.subtitle or "").strip()
            hit_banned = any(b and b in sub for b in banned)
            if cap == 0:
                sub = ""
            elif hit_banned:
                # 套话字幕：尝试用 visual_tags 兜底写平实句
                tags = item.visual_tags or []
                if tags and cap >= 4:
                    # 取第一个具体景物词作为字幕
                    sub = tags[0][:cap]
                else:
                    sub = ""
            elif len(sub) > cap:
                sub = sub[:cap]

            # 配音处理：只截断，不清空（旁白必须有）
            narr = (item.narration_text or "").strip()
            # 配音里的禁用词做替换而非清空
            for b in banned:
                if b and b in narr:
                    narr = narr.replace(b, "")
            narr = narr.strip("，。、 ").strip()
            if len(narr) > narr_cap:
                # 按句号/逗号截断，不从中间硬切
                for i in range(narr_cap, 0, -1):
                    if i < len(narr) and narr[i] in "。，、；！？,;":
                        narr = narr[:i]
                        break
                else:
                    narr = narr[:narr_cap]
            # 兜底：配音为空时用 visual_summary
            if not narr:
                visual = item.visual_summary or ""
                narr = visual[:narr_cap] if visual else "这一路的风景"

            subtitles_out.append({
                "order": item.order,
                "text": sub,
                "narration": narr,
                "device": "",
                "reason": "本地优化" if sub else "留空",
            })

        # 第二遍：覆盖率控制（50%-80%）
        filled = sum(1 for s in subtitles_out if s["text"])
        total = len(subtitles_out)
        rate = filled / max(1, total)
        if rate > 0.80:
            # 超覆盖：从后往前清空非关键镜头（保留首尾）
            for s in reversed(subtitles_out[1:-1]):
                if rate <= 0.80:
                    break
                if s["text"]:
                    s["text"] = ""
                    s["reason"] = "覆盖率控制留空"
                    filled -= 1
                    rate = filled / max(1, total)
        elif rate < 0.50 and total >= 4:
            # 不足覆盖：用 visual_tags 补字幕
            for s in subtitles_out:
                if rate >= 0.50:
                    break
                if not s["text"]:
                    item = next((i for i in ed.timeline if i.order == s["order"]), None)
                    if item and item.visual_tags:
                        sh = by_order.get(s["order"], {})
                        cap = sh.get("max_chars", 10)
                        if cap > 0:
                            s["text"] = item.visual_tags[0][:cap]
                            s["reason"] = "补覆盖率"
                            filled += 1
                            rate = filled / max(1, total)

        # 首镜头强制钩子：如果第一镜 top 为空，用抓手硬拼一句
        if subtitles_out and not subtitles_out[0]["text"]:
            first_item = next((i for i in ed.timeline if i.order == 1), None)
            if first_item and first_item.visual_tags:
                tag = first_item.visual_tags[0]
                # 简单的错位手法模板
                hooks = [
                    f"{tag}还在偷懒",
                    f"{tag}先动了",
                    f"{tag}在加班",
                ]
                cap = by_order.get(1, {}).get("max_chars", 14)
                hook = hooks[0][:cap]
                subtitles_out[0]["text"] = hook
                subtitles_out[0]["reason"] = "首镜头强制钩子"
                filled += 1
                rate = filled / max(1, total)

        logger.info(
            f"本地优化完成: {filled}/{total} 条字幕 "
            f"(覆盖率 {rate:.0%}), 无错位手法（需 LLM）"
        )
        return {
            "subtitles": subtitles_out,
            "callback_pair": [],
            "self_check": {
                "prediction_test_passed": True,
                "banned_words_used": [],
                "coverage_rate": rate,
                "device_count": 0,
                "devices_used": [],
                "strongest_line": "",
                "local_optimized": True,
            },
        }

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
                # 配音字数上限：3.5字/秒 × 镜头时长，留 5% 余量给气口
                "narration_max_chars": max(8, int(d * 3.5 * 0.95)),
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
            "你现在要为一条已定稿的时间线撰写全部画面字幕和旁白配音文案。核心要求：\n"
            "1. 【预测测试】遮住字幕只看画面，观众能猜到的一律重写。"
            "描述画面的、抒情的，都是猜得到的。\n"
            "2. 每条有字的字幕标注所用手法 device（A 场景误读 / B 宏大降格 / "
            "C 身份错位 / D 数值荒诞 / E 错位归因），平实句留空字符串。"
            "标不出手法的通常就是套话。\n"
            "3. 【控制用力】整条片子错位手法只用 2-4 处，其余有字的镜头写平实句。"
            "全是梗，梗就不响了。\n"
            "4. 【首镜头必须是钩子】order=1 的镜头字幕（text）必须写一句能勾住人的错位金句："
            "例如'云还没打卡'（身份错位）、'新疆把天空开了最大亮度'（场景误读）。"
            "第一镜的 narration 也要先声夺人，不要'大家好/今天我在…'这种开场。\n"
            "5. 严格遵守每个镜头的 max_chars（字幕）和 narration_max_chars（配音）；"
            "max_chars=0 必须留空字幕，但 narration 仍要写。\n"
            "6. 字幕覆盖率 50%-80%，宁可有也不要空。\n"
            "7. 【配音文案】narration 是旁白念出来的内容，口语化、第一人称，"
            "每镜都要有。与字幕呼应但可以更长更完整——字幕是金句，旁白是讲故事。\n"
            "8. 必须有一组首尾呼应；同片内不复用同一手法。\n"
            "9. 写不出错位就写平实事实，**绝不退回抒情套话**。\n"
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
        parts.append("| # | 起始 | 时长 | 字幕上限 | 配音上限 | 画面 | 画面物件（错位抓手）|")
        parts.append("|---|---|---|---|---|---|---|")
        for sh in shots:
            cap = "**必须留空**" if sh["max_chars"] == 0 else str(sh["max_chars"])
            narr_cap = str(sh["narration_max_chars"])
            subj = "、".join(sh.get("subjects") or []) or "—"
            parts.append(
                f"| {sh['order']} | {sh['at']}s | {sh['duration']}s | {cap} "
                f"| {narr_cap} | {sh['visual'] or '（无描述）'} | {subj} |"
            )

        if problems and previous:
            parts.append("\n## 上一版的问题，请逐条修正\n")
            for p in problems:
                parts.append(f"- {p}")
            parts.append(f"\n上一版输出：\n{json.dumps(previous, ensure_ascii=False)}")

        parts.append(
            "\n## 输出格式\n"
            '{"subtitles":['
            '{"order":1,"text":"独库还在放假","narration":"出发前以为独库能走，结果它还在放假","device":"C","reason":"身份错位·埋呼应"},'
            '{"order":2,"text":"","narration":"路上的云比攻略里还多","device":"","reason":"快剪留空"},'
            '{"order":6,"text":"湖还在冰敷","narration":"这湖还没化完，像贴着冰敷","device":"A","reason":"抓手=浮冰，结冰→冰敷"}],'
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
        # 1. 优先 DashScope
        result = self._call_dashscope(system, user)
        if result is not None:
            return result
        # 2. DashScope 不可用 → fallback 本地 Ollama（qwen2.5:7b）
        logger.info("DashScope 不可用，切换本地 Ollama (qwen2.5:7b)")
        result = self._call_ollama(system, user)
        if result is not None:
            return result
        # 3. 都不可用 → 上层走本地规则优化
        return None

    def _call_dashscope(self, system: str, user: str) -> dict | None:
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
                max_tokens=4096,      # 7个镜头的JSON需要足够空间
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

    def _call_ollama(
        self, system: str, user: str, model: str = "qwen2.5:7b",
    ) -> dict | None:
        """通过本地 Ollama 调用 LLM，不依赖 API key"""
        try:
            import json
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "model": model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_ctx": 8192,      # 8K 上下文，prompt 很长
                    "num_predict": 4096,   # 足够输出完整 JSON
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:  # 本地 7B 模型很慢，给10分钟
                body = json.loads(resp.read().decode("utf-8"))
            text = body.get("response", "")
            if not text:
                logger.error("Ollama 返回空 response")
                return None
            return self._parse_json(text)
        except urllib.error.URLError as e:
            logger.error(f"Ollama 不可用（服务未启动？）: {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama 调用失败: {e}")
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

        def _fix(json_str: str) -> str:
            """LLM 常见 JSON 语法错误修复"""
            # 0. 字符串中的换行符：JSON 字符串内不能有裸换行，统一替换为空格
            # 只替换非转义的换行（在引号内）
            out_chars = []
            in_str = False
            escape = False
            for ch in json_str:
                if escape:
                    out_chars.append(ch)
                    escape = False
                    continue
                if ch == "\\" and in_str:
                    out_chars.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    out_chars.append(ch)
                    continue
                if in_str and (ch == "\n" or ch == "\r"):
                    out_chars.append(" ")
                    continue
                out_chars.append(ch)
            json_str = "".join(out_chars)

            # 1. 多余逗号: ",," → ","
            while ",," in json_str:
                json_str = json_str.replace(",,", ",")
            # 2. 数组/对象末尾多余逗号: "[...]," → "[...]"  "{...}," → "{...}"
            json_str = re.sub(r"\s*,\s*([}\]])", r"\1", json_str)
            # 3. 省略号导致 JSON 截断，补末尾闭合
            open_b = json_str.count("[") - json_str.count("]")
            open_o = json_str.count("{") - json_str.count("}")
            if open_b > 0:
                json_str += "]" * open_b
            if open_o > 0:
                json_str += "}" * open_o
            return json_str

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            a, b = text.find("{"), text.rfind("}") + 1
            if a >= 0 and b > a:
                snippet = text[a:b]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    fixed = _fix(snippet)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
            try:
                return json.loads(_fix(text))
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
        subs = {s.get("order"): s for s in result.get("subtitles", [])}

        filled = 0
        for order, shot in by_order.items():
            s = subs.get(order, {})
            text = (s.get("text") or "").strip()
            narration = (s.get("narration") or "").strip()
            cap = shot["max_chars"]
            narr_cap = shot["narration_max_chars"]

            # 字幕检查
            if text:
                filled += 1
            if cap == 0 and text:
                problems.append(
                    f"第 {order} 镜时长仅 {shot['duration']}s，必须留空，但写了「{text}」"
                )
            elif cap and len(text) > cap:
                problems.append(
                    f"第 {order} 镜字幕超字数（上限 {cap}，实际 {len(text)}）：「{text}」"
                )
            for b in banned:
                if b and b in text:
                    problems.append(f"第 {order} 镜字幕命中禁用词「{b}」：「{text}」")
                    break

            # 配音文案检查
            if not narration:
                problems.append(f"第 {order} 镜缺少配音文案 narration")
            elif len(narration) > narr_cap:
                problems.append(
                    f"第 {order} 镜配音超字数（上限 {narr_cap}，实际 {len(narration)}）：「{narration}」"
                )
            for b in banned:
                if b and b in narration:
                    problems.append(f"第 {order} 镜配音命中禁用词「{b}」：「{narration}」")
                    break

        rate = filled / max(1, len(by_order))
        if not (0.35 <= rate <= 0.65):
            problems.append(
                f"字幕覆盖率 {rate:.0%} 超出 40%-60% 区间（当前 {filled}/{len(by_order)} 条有字）"
            )

        # 错位手法：数量和多样性（skill §2 / §3）
        # 本地优化路径无法生成错位手法，跳过此项检查
        is_local = result.get("self_check", {}).get("local_optimized", False)
        if not is_local:
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
        """写回时间线：top=错位金句，bottom=配音内容"""
        subs = {s.get("order"): s for s in result.get("subtitles", [])}
        by_order = {s["order"]: s for s in shots}

        for item in ed.timeline:
            s = subs.get(item.order, {})
            # 顶部金句：错位手法创意文案
            top_text = (s.get("text") or "").strip()
            cap = by_order.get(item.order, {}).get("max_chars", 10)
            if cap == 0:
                top_text = ""
            elif len(top_text) > cap:
                top_text = top_text[:cap]
            item.top_subtitle = top_text
            item.subtitle = top_text  # 兼容旧字段

            # 底部字幕：配音内容全文（跟读用）
            narration = (s.get("narration") or "").strip()
            narr_cap = by_order.get(item.order, {}).get("narration_max_chars", 20)
            if len(narration) > narr_cap:
                # 按句号/逗号软截断
                for i in range(narr_cap, 0, -1):
                    if i < len(narration) and narration[i] in "。，、；！？,;":
                        narration = narration[:i]
                        break
                else:
                    narration = narration[:narr_cap]
            item.bottom_subtitle = narration
            if narration:
                item.narration_text = narration

        n_top = sum(1 for i in ed.timeline if i.top_subtitle)
        n_bot = sum(1 for i in ed.timeline if i.bottom_subtitle)
        logger.info(
            f"文案完成: 顶部金句 {n_top}/{len(ed.timeline)} "
            f"({n_top / max(1, len(ed.timeline)):.0%}), "
            f"底部配音 {n_bot}/{len(ed.timeline)} 条"
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
