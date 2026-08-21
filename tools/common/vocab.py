"""受控词表 — VLM 打标输出 / DB 结构化过滤 / skills 约束三者共用

这是把「知识层的规范」和「数据层的字段」对齐的地基。
skills/travel-pacing.md 的动静结合、景别嵌套，以及
skills/travel-copywriting.md 手法 A 的物件抓手，都依赖这里的枚举。
"""

from __future__ import annotations

# ── 景别（skills/travel-pacing.md 景别嵌套约束用）──
SHOT_SCALE = ["ELS", "LS", "MS", "CU", "ECU"]
SHOT_SCALE_CN = {
    "ELS": "大远景（航拍/超广，展示整体环境）",
    "LS": "远景（全身+环境）",
    "MS": "中景（腰部以上）",
    "CU": "近景（胸部以上或物体特写）",
    "ECU": "大特写（极致局部）",
}

# ── 运镜（动静结合约束用）──
CAMERA_MOTION = [
    "static", "push_in", "pull_out",      # 归为「静」
    "pan", "tilt", "tracking", "handheld", "aerial",  # 归为「动」
]
CAMERA_MOTION_CN = {
    "static": "固定机位", "push_in": "推近", "pull_out": "拉远",
    "pan": "水平摇", "tilt": "垂直摇", "tracking": "跟拍/移动",
    "handheld": "手持晃动", "aerial": "航拍飞行",
}
STATIC_MOTIONS = {"static", "push_in", "pull_out"}


def motion_class(camera_motion: str) -> str:
    """归一到动/静二分，供 pacing 约束直接过滤"""
    return "static" if camera_motion in STATIC_MOTIONS else "dynamic"


# ── 时段光线（旅行素材价值极高，且文案手法 C/E 的原料）──
TIME_OF_DAY = ["sunrise", "golden_hour", "midday", "overcast",
               "blue_hour", "night", "indoor", "unknown"]
TIME_OF_DAY_CN = {
    "sunrise": "日出", "golden_hour": "金色时刻", "midday": "正午",
    "overcast": "阴天", "blue_hour": "蓝调时刻", "night": "夜景",
    "indoor": "室内", "unknown": "无法判断",
}

WEATHER = ["晴", "多云", "阴", "雨", "雪", "雾", "unknown"]

MOOD = ["epic", "calm", "lively", "curious", "melancholic", "cheerful", "neutral"]

DEFECT = ["过曝", "欠曝", "失焦", "严重抖动", "穿帮", "画面遮挡", "画质差"]


# ── 20 类场景分类 ──
# 保留，但**降级为附加索引**（用于结构化预过滤），不再是打标的唯一输出。
# 之前的问题：一段视频被压成 1 个枚举 + 一组查表词，
# 具体物件信息在入库时就丢了，导致检索粒度过粗、文案抓不出手法。
SCENE_CATEGORY = [
    "snow_mountain", "grassland", "lake", "river", "forest", "canyon",
    "desert", "sea", "road", "aerial", "sunset", "starry_sky", "sky",
    "architecture", "city", "flower", "reflection", "people", "food",
    "animal", "other",
]
SCENE_CATEGORY_CN = {
    "snow_mountain": "雪山山脉", "grassland": "草原田野", "lake": "湖泊水景",
    "river": "河流瀑布", "forest": "森林树木", "canyon": "峡谷地貌",
    "desert": "沙漠戈壁", "sea": "大海海滩", "road": "公路自驾",
    "aerial": "航拍俯视", "sunset": "日落日出", "starry_sky": "星空银河",
    "sky": "天空云海", "architecture": "建筑人文", "city": "城市风光",
    "flower": "花海花朵", "reflection": "倒影镜面", "people": "人物人像",
    "food": "美食小吃", "animal": "野生动物", "other": "其他",
}


def normalize_enum(value: str, allowed: list[str], default: str) -> str:
    """把模型输出规整到受控词表内，防止脏值进库"""
    v = (value or "").strip()
    if v in allowed:
        return v
    low = v.lower()
    for a in allowed:
        if a.lower() == low:
            return a
    return default
