"""CLIP 零样本场景分类器 — 免费本地运行，无需 API key

使用 HuggingFace transformers 的 CLIP 模型，
对视频帧进行零样本分类，识别 20 种旅行场景类型。
同时生成 top-k 视觉标签。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from tools.common.config import CACHE_DIR

logger = logging.getLogger(__name__)

# 20 种旅行场景分类 + 英文描述（用于 CLIP 零样本分类）
# 描述要具体、有区分度，避免类别之间过于相似
TRAVEL_SCENE_CLASSES = {
    "snow_mountain": "snowy mountain peaks with white snow, alpine landscape, high altitude mountains",
    "lake": "calm blue lake with mountain reflection, peaceful water landscape",
    "river": "flowing river or waterfall in a valley, stream with moving water",
    "sea": "ocean beach with waves and sand, coastal seascape, blue sea",
    "grassland": "vast green grassland prairie with grazing animals, open meadow",
    "forest": "dense green forest with tall pine trees, woodland scenery",
    "desert": "sandy desert with sand dunes under hot sun, arid gobi landscape",
    "canyon": "rocky canyon with steep cliffs, mountain gorge landscape",
    "road": "winding highway road through mountains, road trip view from car",
    "sunset": "golden sunset over landscape, orange sky with warm sunlight",
    "starry_sky": "night sky with milky way and stars, dark starry landscape",
    "sky": "blue sky with white clouds, aerial view of cloud sea",
    "flower": "field of colorful flowers in bloom, beautiful flower garden",
    "architecture": "traditional temple or ancient building, historic architecture",
    "city": "modern city skyline with skyscrapers, urban cityscape",
    "food": "delicious local food dishes on table, meal with various dishes",
    "people": "person or people traveling, tourist taking photos, portrait",
    "animal": "wild animals in nature, birds or horses in landscape",
    "aerial": "drone aerial view from above, bird's eye view of landscape",
    "reflection": "perfect water reflection of mountains and sky, mirror lake",
}

# 分类对应的中文显示名
CATEGORY_DISPLAY_NAMES = {
    "snow_mountain": "雪山山脉",
    "lake": "湖泊水景",
    "river": "河流瀑布",
    "sea": "大海海滩",
    "grassland": "草原田野",
    "forest": "森林树木",
    "desert": "沙漠戈壁",
    "canyon": "峡谷地貌",
    "road": "公路自驾",
    "sunset": "日落日出",
    "starry_sky": "星空银河",
    "sky": "天空云海",
    "flower": "花海花朵",
    "architecture": "建筑人文",
    "city": "城市风光",
    "food": "美食小吃",
    "people": "人物人像",
    "animal": "野生动物",
    "aerial": "航拍俯视",
    "reflection": "倒影镜面",
}

# 详细视觉标签（用于补充描述）
VISUAL_TAG_CANDIDATES = {
    "snow_mountain": ["雪山", "冰川", "雪峰", "高山", "壮观", "雄伟", "白", "天空"],
    "lake": ["湖泊", "湖水", "湛蓝", "平静", "倒影", "雪山", "蓝天", "清澈"],
    "river": ["河流", "瀑布", "溪流", "江水", "清澈", "峡谷", "流动", "碧绿"],
    "sea": ["大海", "海洋", "海滩", "海浪", "沙滩", "蔚蓝", "海岸", "浪花"],
    "grassland": ["草原", "草地", "牛羊", "绿色", "广阔", "牧", "马匹", "蓝天"],
    "forest": ["森林", "树木", "云杉", "松树", "绿色", "光影", "幽静", "树林"],
    "desert": ["沙漠", "戈壁", "沙丘", "黄沙", "荒凉", "日落", "骆驼", "干旱"],
    "canyon": ["峡谷", "悬崖", "峭壁", "岩石", "地貌", "壮观", "深红", "河流"],
    "road": ["公路", "自驾", "沿途", "风景", "弯道", "在路上", "车窗", "远方"],
    "sunset": ["日落", "夕阳", "晚霞", "金色", "黄昏", "橙色", "暖光", "剪影"],
    "starry_sky": ["星空", "银河", "夜景", "星星", "夜空", "寂静", "月光", "星轨"],
    "sky": ["天空", "云海", "蓝天", "白云", "航拍", "辽阔", "阳光", "全景"],
    "flower": ["花海", "花朵", "野花", "绽放", "美丽", "粉色", "春天", "花园"],
    "architecture": ["建筑", "人文", "寺庙", "古城", "村落", "传统", "文化", "历史"],
    "city": ["城市", "都市", "繁华", "高楼", "地标", "街道", "夜景", "现代"],
    "food": ["美食", "当地", "特色", "味道", "好吃", "小吃", "菜品", "食欲"],
    "people": ["人物", "人像", "自拍", "旅行", "朋友", "笑容", "记录", "生活"],
    "animal": ["动物", "野生动物", "自然", "生态", "可爱", "鸟", "马", "牛"],
    "aerial": ["航拍", "俯视", "鸟瞰", "全景", "高空", "壮观", "上帝视角", "无人机"],
    "reflection": ["倒影", "镜面", "对称", "宁静", "美丽", "湖水", "天空", "平静"],
}


class CLIPZeroShotClassifier:
    """CLIP 零样本分类器（基于 open_clip）

    用法：
        clf = CLIPZeroShotClassifier()
        result = clf.classify_frame("path/to/frame.jpg")
        # {"category": "lake", "confidence": 0.85, "top_tags": [...]}
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        self._class_names = list(TRAVEL_SCENE_CLASSES.keys())
        self._class_descriptions = list(TRAVEL_SCENE_CLASSES.values())

    def _ensure_model(self):
        """延迟加载模型（首次使用时下载）"""
        if self._model is not None:
            return

        logger.info(f"加载 CLIP 模型: {self.model_name} ({self.pretrained})")
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=device,
        )
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

        # 预热 text embeddings
        model.eval()
        with torch.no_grad():
            text = self._tokenizer(self._class_descriptions).to(device)
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self._text_features = text_features

        logger.info(f"CLIP 模型加载完成 (device={device})")

    def classify_frame(self, frame_path: str, top_k: int = 5) -> dict[str, Any]:
        """对单帧图片进行分类

        Returns:
            {
                "category": "lake",         # 最可能的类别
                "confidence": 0.85,         # 置信度
                "top_categories": [...],    # top-k 类别
                "top_tags": [...],          # 推荐视觉标签
            }
        """
        try:
            self._ensure_model()
        except Exception as e:
            logger.warning(f"CLIP 模型加载失败: {e}")
            return {
                "category": "other",
                "confidence": 0.0,
                "top_categories": [],
                "top_tags": ["风景", "旅行"],
            }

        import torch
        from PIL import Image

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            logger.warning(f"图片读取失败: {e}")
            return {
                "category": "other",
                "confidence": 0.0,
                "top_categories": [],
                "top_tags": ["风景", "旅行"],
            }

        try:
            image_input = self._preprocess(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ self._text_features.T).squeeze(0)
                probs = similarity.softmax(dim=-1).cpu().tolist()
        except Exception as e:
            logger.warning(f"CLIP 推理失败: {e}")
            return {
                "category": "other",
                "confidence": 0.0,
                "top_categories": [],
                "top_tags": ["风景", "旅行"],
            }

        # 按置信度排序
        indexed = list(zip(self._class_names, probs))
        indexed.sort(key=lambda x: -x[1])

        top_categories = [
            {"category": cat, "confidence": round(prob, 3)}
            for cat, prob in indexed[:top_k]
        ]

        best_cat = indexed[0][0]
        best_conf = indexed[0][1]

        # 从 top-3 类别中混合生成视觉标签
        top_tags = self._generate_tags(indexed[:3])

        return {
            "category": best_cat,
            "confidence": round(best_conf, 3),
            "top_categories": top_categories,
            "top_tags": top_tags,
        }

    def _generate_tags(self, top_categories: list[tuple[str, float]]) -> list[str]:
        """根据 top-k 类别生成混合视觉标签"""
        tags = []
        total_conf = sum(c[1] for c in top_categories)

        for cat, conf in top_categories:
            if cat in VISUAL_TAG_CANDIDATES:
                # 按置信度比例取标签
                n_tags = max(2, int(6 * conf / total_conf))
                cat_tags = VISUAL_TAG_CANDIDATES[cat][:n_tags]
                for t in cat_tags:
                    if t not in tags:
                        tags.append(t)

        # 保证至少 4 个标签
        if len(tags) < 4 and top_categories:
            best_cat = top_categories[0][0]
            if best_cat in VISUAL_TAG_CANDIDATES:
                for t in VISUAL_TAG_CANDIDATES[best_cat]:
                    if t not in tags:
                        tags.append(t)
                    if len(tags) >= 6:
                        break

        return tags[:6]

    def encode_image(self, image_path: str) -> list[float] | None:
        """将图像编码为 CLIP 向量"""
        self._ensure_model()
        if self._model is None:
            return None

        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"图片读取失败: {e}")
            return None

        try:
            import torch
            image_input = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.squeeze(0).cpu().tolist()
        except Exception as e:
            logger.warning(f"图像编码失败: {e}")
            return None

    def encode_text(self, text: str) -> list[float] | None:
        """将文本编码为 CLIP 向量"""
        self._ensure_model()
        if self._model is None:
            return None

        try:
            import torch
            text_tokens = self._tokenizer([text]).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu().tolist()
        except Exception as e:
            logger.warning(f"文本编码失败: {e}")
            return None


# ── 视频帧提取工具 ──────────────────────────────────────────

def extract_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    """从视频中提取指定时间的帧"""
    try:
        subprocess.run(
            [
                "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
                "-frames:v", "1", "-q:v", "3",
                "-vf", "scale=512:-1",
                output_path, "-y",
            ],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.debug(f"抽帧失败: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0
