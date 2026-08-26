"""风格注册表：humor（幽默吐槽）/ real（活人感真实）/ contrast（有反差对比）

三种风格各覆盖：
- name:         风格标识（= style variable，写进文件名）
- label:        中文标签（显示用）
- persona:      copywriter.write 里的 CreativeBrief.persona
- avoid_words:  禁用词，会并入 CreativeBrief.avoid + copywriter 的 banned 合并清单
- hook_priority:  优先的 hook_type 顺序（script_generator 选 hook 时加权）
- device_priority:  优先用的错位手法顺序（copywriter 里 A-E），会在 prompt 里提示
- narrative:    叙事偏好（短/长句？开头钩子放哪？首尾呼应怎么做？）
- tts_voice:    TTS 音色
- tts_rate:     TTS 语速
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StyleProfile:
    name: str
    label: str
    persona: str
    avoid_words: list[str] = field(default_factory=list)
    hook_priority: list[str] = field(default_factory=list)
    device_priority: list[str] = field(default_factory=list)
    narrative: str = ""
    tts_voice: str = "yunjian"
    tts_rate: str = "-5%"


STYLES: dict[str, StyleProfile] = {
    # ──────────────────────────────────────────────────
    # 幽默吐槽风：沙雕朋友视角，碎碎念、自嘲、反差梗
    # ──────────────────────────────────────────────────
    "humor": StyleProfile(
        name="humor",
        label="幽默吐槽",
        persona=(
            "朋友吐槽视角：沙雕、碎碎念，爱自嘲，爱给风景贴不正经的人设，"
            "爱用'这家伙''这货''好家伙'，偶尔自黑钱包/体力/拍照技术不行。"
            "像跟损友现场汇报，不要官方腔。"
        ),
        avoid_words=[
            "美不胜收", "心灵的故乡", "治愈", "净化灵魂", "心灵净土",
            "远离喧嚣", "人间仙境", "世外桃源", "一生必去", "洗涤心灵",
            "诗和远方",
        ],
        hook_priority=["number", "question", "shock", "contrast"],
        device_priority=["C", "A", "D", "B", "E"],  # 身份错位/场景误读优先
        narrative=(
            "前2秒必须抛梗：把景点拟人化、把狼狈事说出来；中间段落 2-3 处错位手法；"
            "结尾可以自嘲或调侃钱包/体重，首尾呼应一组'狼狈→释然'的情绪弧。"
            "台词短句子多，每句尽量有包袱。"
        ),
        tts_voice="yunxia",  # 云夏（男声，略带跳脱年轻感，比沉稳的云健更适合吐槽）
        tts_rate="+5%",      # 稍快，吐槽节奏
    ),

    # ──────────────────────────────────────────────────
    # 活人感真实风：第一人称亲历者，写狼狈/意外/真实小细节
    # ──────────────────────────────────────────────────
    "real": StyleProfile(
        name="real",
        label="活人感真实",
        persona=(
            "真人亲历视角：克制、不煽情、像跟朋友讲事。写具体的小狼狈、小意外、小确幸——"
            "'没赶上车''鞋子进了沙''多花了80块''太阳比攻略里毒'这种真实细节。"
            "不写大词，不升华，不抒情。像在跟朋友转述一天发生的事。"
        ),
        avoid_words=[
            "震撼", "心灵的震撼", "此生难忘", "岁月静好", "一眼万年",
            "神往", "最美的风景", "天堂", "圣地", "邂逅",
            "洗涤心灵", "诗和远方",
        ],
        hook_priority=["resonance", "number", "question"],
        device_priority=["E", "C", "A", "B", "D"],  # 错位归因（把小意外甩锅给风景）优先
        narrative=(
            "开头抛一个真实小狼狈（'新疆第一天我就被骗了'）；中间每镜写一个具体事实；"
            "结尾不要升华，用一个普通的小画面收——'日落的时候蚊子有点多'。"
            "首尾呼应一组'期待 vs 真实'的小反差。"
        ),
        tts_voice="yunjian",  # 云健：沉稳男声，讲故事更真实
        tts_rate="-8%",       # 稍慢，像娓娓道来
    ),

    # ──────────────────────────────────────────────────
    # 反差对比风：期待 vs 现实 / 宏大 vs 渺小 / 热闹 vs 孤独
    # ──────────────────────────────────────────────────
    "contrast": StyleProfile(
        name="contrast",
        label="反差对比",
        persona=(
            "观察派：冷静、镜头感强、爱写对比。同一镜头里找成对的反义——"
            "攻略 vs 现场、人山人海 vs 角落一人、雪山的冷 vs 奶茶的热、4000km 路程 vs 1 分钟感动。"
            "一句话里必须塞两个对立面，不写单独的褒或贬。"
        ),
        avoid_words=[
            "美如画", "风景如画", "如诗如画", "美不胜收", "心灵的故乡",
            "治愈", "净化灵魂", "人间仙境", "一生必去", "诗和远方",
        ],
        hook_priority=["contrast", "shock", "number"],
        device_priority=["B", "E", "D", "A", "C"],  # 宏大降格、错位归因优先
        narrative=(
            "开头必须抛一组强对比（'攻略叫我穿羽绒服，我在新疆短袖出了汗'）；"
            "中间每镜至少带一个对照词——但/却/竟然/vs——把两个极端捆在一起；"
            "结尾用一句话把片子的对比收住。"
            "错位手法集中在 B（宏大降格）和 D（数值荒诞）。"
        ),
        tts_voice="yunjian",  # 云健：冷静男声，更适合观察派对比口吻
        tts_rate="-5%",
    ),
}


DEFAULT_STYLE_ORDER = ["humor", "real", "contrast"]


def get_style(name: str) -> StyleProfile:
    if name not in STYLES:
        raise KeyError(
            f"未知风格 '{name}'。可选：{', '.join(STYLES.keys())}"
        )
    return STYLES[name]
