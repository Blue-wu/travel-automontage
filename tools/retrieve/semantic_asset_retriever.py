"""语义检索模块 — 基于 CLIP 向量的素材语义检索

参考 OpenMontage 的 Documentary Montage 做法：从素材库用 CLIP 语义检索。
区别是 OpenMontage 从 Pexels/Archive.org 检索公开素材，
我们从"自有素材库"检索——素材独特性 = 内容独特性。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tools.common.asset_store import AssetStore
from tools.common.model_client import ModelClient
from tools.common.models import CandidateClip, CandidateClips, SceneCategory

logger = logging.getLogger(__name__)

SCENE_CATEGORIES = {
    "snow_mountain": {
        "display_name": "雪山山脉",
        "keywords": ["雪山", "山", "峰", "冰川", "雪", "山脉", "高", "雄伟", "壮观", "雪山山脉"],
    },
    "lake": {
        "display_name": "湖泊水景",
        "keywords": ["湖", "湖泊", "水", "蓝湖", "天池", "倒影", "湛蓝", "湖水"],
    },
    "river": {
        "display_name": "河流瀑布",
        "keywords": ["河", "河流", "溪", "溪流", "瀑布", "江水", "河谷", "峡谷河流"],
    },
    "sea": {
        "display_name": "大海海滩",
        "keywords": ["海", "海洋", "大海", "海滩", "沙滩", "海边", "海浪", "海岸"],
    },
    "grassland": {
        "display_name": "草原田野",
        "keywords": ["草原", "草地", "田", "牧场", "牛", "羊", "马", "绿", "牧", "大草原"],
    },
    "forest": {
        "display_name": "森林树木",
        "keywords": ["森林", "树", "林", "云杉", "松", "木", "林荫", "绿色", "树林"],
    },
    "desert": {
        "display_name": "沙漠戈壁",
        "keywords": ["沙漠", "戈壁", "沙丘", "荒漠", "沙", "黄沙", "雅丹"],
    },
    "canyon": {
        "display_name": "峡谷地貌",
        "keywords": ["峡谷", "悬崖", "峭壁", "丹霞", "喀斯特", "地貌", "岩石"],
    },
    "road": {
        "display_name": "公路自驾",
        "keywords": ["公路", "路", "开车", "驾驶", "沿途", "路标", "弯道", "车内", "窗外", "自驾"],
    },
    "sunset": {
        "display_name": "日落日出",
        "keywords": ["日落", "夕阳", "黄昏", "晚霞", "金色", "橙色", "暖", "光影", "日出", "朝阳"],
    },
    "starry_sky": {
        "display_name": "星空银河",
        "keywords": ["星空", "银河", "夜景", "星星", "夜空", "黑", "月光", "月亮", "星轨"],
    },
    "sky": {
        "display_name": "天空云海",
        "keywords": ["天空", "云", "蓝天", "云海", "晴", "阳光", "光", "全景", "云彩"],
    },
    "flower": {
        "display_name": "花海花朵",
        "keywords": ["花", "花海", "花朵", "野花", "花季", "桃花", "油菜花", "花园"],
    },
    "architecture": {
        "display_name": "建筑人文",
        "keywords": ["建筑", "房子", "村", "镇", "街", "寺庙", "古迹", "人文", "古城", "村落"],
    },
    "city": {
        "display_name": "城市风光",
        "keywords": ["城市", "都市", "街道", "高楼", "夜景", "繁华", "地标"],
    },
    "food": {
        "display_name": "美食小吃",
        "keywords": ["美食", "吃", "菜", "饭", "面", "肉", "小吃", "味道", "当地美食", "特色菜"],
    },
    "people": {
        "display_name": "人物人像",
        "keywords": ["人", "人物", "自拍", "人像", "朋友", "合影", "当地人", "游客"],
    },
    "animal": {
        "display_name": "野生动物",
        "keywords": ["动物", "鸟", "马", "牛", "羊", "骆驼", "野生动物", "候鸟"],
    },
    "aerial": {
        "display_name": "航拍俯视",
        "keywords": ["航拍", "俯视", "鸟瞰", "无人机", "高空", "上帝视角", "全景航拍"],
    },
    "reflection": {
        "display_name": "倒影镜面",
        "keywords": ["倒影", "镜面", "对称", "水中倒影", "天空之镜"],
    },
}


class SemanticAssetRetriever:
    """语义素材检索器

    用法：
        retriever = SemanticAssetRetriever()
        results = retriever.retrieve(
            query="富士山日出 金色时刻 震撼全景",
            destination="日本-富士山",
            top_k=20
        )
    """

    def __init__(
        self,
        asset_store: AssetStore | None = None,
        model_client: ModelClient | None = None,
    ):
        self.asset_store = asset_store or AssetStore()
        self.model_client = model_client or ModelClient()

    def retrieve(
        self,
        query: str,
        destination: str | None = None,
        top_k: int = 20,
        min_quality: float = 0.0,
    ) -> CandidateClips:
        """语义检索素材

        Args:
            query: 查询语句，如 "富士山日出 金色时刻 震撼全景"
            destination: 目的地过滤（可选）
            top_k: 返回 Top-K 结果
            min_quality: 最低质量分数过滤

        Returns:
            CandidateClips: 候选素材片段列表
        """
        logger.info(f"语义检索: query='{query}', destination={destination}, top_k={top_k}")

        # Step 1: 向量化查询
        query_embedding = self.model_client.embed_text(query)
        if query_embedding is None:
            logger.warning("CLIP 不可用，使用关键词匹配回退")
            return self._fallback_keyword_search(query, destination, top_k, min_quality)

        # Step 2: 获取素材库中的所有场景向量
        all_vectors = self.asset_store.get_all_embeddings(destination)
        if not all_vectors:
            logger.warning("素材库为空或无向量数据")
            return CandidateClips(query=query, destination=destination or "", clips=[])

        # Step 3: 计算余弦相似度
        query_vec = np.array(query_embedding, dtype=np.float32)
        scored = []
        for asset_id, scene_index, summary, embedding in all_vectors:
            if not embedding:
                continue

            # 获取素材质量分数
            asset = self.asset_store.get(asset_id)
            quality = 0.0
            source_path = ""
            start_sec = 0.0
            end_sec = 10.0
            visual_tags = []

            scene_category = "other"
            subjects = []
            if asset and scene_index < len(asset.scenes):
                scene = asset.scenes[scene_index]
                quality = scene.quality
                source_path = asset.source_path
                start_sec = scene.start_sec
                end_sec = scene.end_sec
                visual_tags = scene.visual_tags
                subjects = list(getattr(scene, "subjects", None) or [])
                scene_category = scene.scene_category

            if quality < min_quality:
                continue

            # 余弦相似度
            emb_vec = np.array(embedding, dtype=np.float32)
            sim = float(np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec) + 1e-8
            ))

            # 关键词加权：如果查询词命中标签/描述，额外加分
            kw_boost = 0.0
            query_lower = query.lower()
            tag_text = " ".join(visual_tags).lower() + " " + (summary or "").lower()
            for char in query:
                if char.strip() and char in tag_text:
                    kw_boost += 0.02
            kw_boost = min(kw_boost, 0.15)  # 最多加 0.15

            final_score = sim + kw_boost

            scored.append({
                "asset_id": asset_id,
                "source_path": source_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "score": final_score,
                "scene_summary": summary,
                "visual_tags": visual_tags,
                "subjects": subjects,
                "scene_category": scene_category,
                "quality": quality,
            })

        # Step 4: 多样性 Top-K
        diverse_results = self._diverse_top_k(scored, top_k)

        clips = [
            CandidateClip(**clip_data) for clip_data in diverse_results
        ]

        logger.info(f"检索完成: {len(clips)} 个候选片段")
        return CandidateClips(
            query=query,
            destination=destination or "",
            total_found=len(scored),
            clips=clips,
        )

    def retrieve_v2(
        self,
        query: str,
        destination: str | None = None,
        top_k: int = 20,
        min_quality: float = 0.0,
        clip_weight: float = 0.6,
        video_weight: float = 0.4,
    ) -> CandidateClips:
        """V2 语义检索：CLIP + VideoMAE 双路融合

        同时使用 CLIP 图像特征和 VideoMAE 视频特征进行检索，
        加权融合得到最终分数，兼顾静态语义和动态语义。

        Args:
            query: 查询语句
            destination: 目的地过滤
            top_k: 返回 Top-K 结果
            min_quality: 最低质量分数过滤
            clip_weight: CLIP 相似度权重
            video_weight: VideoMAE 相似度权重

        Returns:
            CandidateClips: 候选素材片段列表
        """
        logger.info(f"V2 语义检索 (双路融合): query='{query}', destination={destination}, top_k={top_k}")

        # Step 1: CLIP 文本向量
        clip_query_embedding = self.model_client.embed_text(query)
        if clip_query_embedding is None:
            logger.warning("CLIP 不可用，回退到普通检索")
            return self.retrieve(query, destination, top_k, min_quality)

        # Step 2: 获取所有双向量
        all_vectors = self.asset_store.get_all_embeddings_v2(destination)
        if not all_vectors:
            logger.warning("素材库为空或无向量数据")
            return CandidateClips(query=query, destination=destination or "", clips=[])

        # 检查是否有 video embedding
        has_video = any(v[4] is not None for v in all_vectors)
        if not has_video:
            logger.info("无 VideoMAE 向量，回退到纯 CLIP 检索")
            return self.retrieve(query, destination, top_k, min_quality)

        # Step 3: 计算双路相似度
        clip_query_vec = np.array(clip_query_embedding, dtype=np.float32)
        scored = []

        for asset_id, scene_index, summary, clip_emb, video_emb in all_vectors:
            if not clip_emb:
                continue

            # 获取素材信息
            asset = self.asset_store.get(asset_id)
            quality = 0.0
            source_path = ""
            start_sec = 0.0
            end_sec = 10.0
            visual_tags = []
            subjects = []
            scene_category = "other"

            if asset and scene_index < len(asset.scenes):
                scene = asset.scenes[scene_index]
                quality = scene.quality
                source_path = asset.source_path
                start_sec = scene.start_sec
                end_sec = scene.end_sec
                visual_tags = scene.visual_tags
                subjects = list(getattr(scene, "subjects", None) or [])
                scene_category = scene.scene_category

            if quality < min_quality:
                continue

            # CLIP 相似度
            clip_vec = np.array(clip_emb, dtype=np.float32)
            clip_sim = float(np.dot(clip_query_vec, clip_vec) / (
                np.linalg.norm(clip_query_vec) * np.linalg.norm(clip_vec) + 1e-8
            ))

            # VideoMAE 相似度（用 CLIP 文本向量做近似检索，
            # 因为 VideoMAE 是视频预训练模型，没有文本对齐，
            # 这里用 CLIP 向量空间做桥梁）
            video_sim = 0.0
            if video_emb is not None:
                video_vec = np.array(video_emb, dtype=np.float32)
                # 跨模态相似度：直接用余弦相似度作为粗粒度关联
                video_sim = float(np.dot(clip_query_vec, video_vec) / (
                    np.linalg.norm(clip_query_vec) * np.linalg.norm(video_vec) + 1e-8
                ))
                # 归一化 VideoMAE 相似度（VideoMAE 与 CLIP 不在同一空间，分数偏低）
                video_sim = (video_sim + 1) / 2  # 映射到 [0, 1]

            # 加权融合
            clip_sim_norm = (clip_sim + 1) / 2  # 映射到 [0, 1]
            final_score = clip_weight * clip_sim_norm + video_weight * video_sim

            scored.append({
                "asset_id": asset_id,
                "source_path": source_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "score": final_score,
                "scene_summary": summary,
                "visual_tags": visual_tags,
                "subjects": subjects,
                "scene_category": scene_category,
                "quality": quality,
            })

        # Step 4: 多样性 Top-K
        diverse_results = self._diverse_top_k(scored, top_k)

        clips = [
            CandidateClip(**clip_data) for clip_data in diverse_results
        ]

        logger.info(f"V2 检索完成: {len(clips)} 个候选片段 (clip_w={clip_weight}, video_w={video_weight})")
        return CandidateClips(
            query=query,
            destination=destination or "",
            total_found=len(scored),
            clips=clips,
        )

    def retrieve_multi(
        self,
        destinations: list[str],
        mood: str | None = None,
        top_k_per_destination: int = 8,
    ) -> CandidateClips:
        """多目的地素材检索（用于混剪）"""
        all_clips: list[CandidateClip] = []
        query = f"旅行混剪 {mood or ''}"

        for dest in destinations:
            query_text = f"{dest} 旅行 {mood or ''}"
            results = self.retrieve(query_text, destination=dest, top_k=top_k_per_destination)
            all_clips.extend(results.clips)

        # 按分数排序
        all_clips.sort(key=lambda c: -c.score)

        return CandidateClips(
            query=query,
            destination="; ".join(destinations),
            total_found=len(all_clips),
            clips=all_clips,
        )

    def _diverse_top_k(
        self,
        scored: list[dict[str, Any]],
        k: int,
        min_diversity: float = 0.15,
    ) -> list[dict[str, Any]]:
        """多样性 Top-K — 避免返回全部相似场景

        贪心算法：按分数排序，依次选择，跳过与已选结果过于相似的。
        """
        if len(scored) <= k:
            return scored

        sorted_items = sorted(scored, key=lambda x: -x["score"])
        result = [sorted_items[0]]

        for item in sorted_items[1:]:
            if len(result) >= k:
                break

            # 检查与已选结果的多样性
            is_diverse = True
            for selected in result:
                # 用 source_path + 时间段判断是否为同一场景
                if (item["source_path"] == selected["source_path"] and
                    abs(item["start_sec"] - selected["start_sec"]) < 5.0):
                    is_diverse = False
                    break

            if is_diverse:
                result.append(item)

        # 如果多样性过滤后不足 k 个，补满
        if len(result) < k:
            for item in sorted_items:
                if item not in result:
                    result.append(item)
                    if len(result) >= k:
                        break

        return result

    def _fallback_keyword_search(
        self,
        query: str,
        destination: str | None,
        top_k: int,
        min_quality: float,
    ) -> CandidateClips:
        """CLIP 不可用时的关键词匹配回退方案"""
        query_words = set(query.lower().split())

        candidates = []
        if destination:
            assets = self.asset_store.filter_by_destination(destination)
        else:
            assets = list(self.asset_store.list_all())

        for asset in assets:
            for i, scene in enumerate(asset.scenes):
                if scene.quality < min_quality:
                    continue

                # 关键词匹配
                scene_tags = set(tag.lower() for tag in scene.visual_tags)
                summary_words = set(scene.summary.lower().split())
                all_words = scene_tags | summary_words

                overlap = len(query_words & all_words)
                score = overlap / max(len(query_words), 1) if query_words else 0.0

                if score > 0 or not query_words:
                    candidates.append({
                        "asset_id": asset.asset_id,
                        "source_path": asset.source_path,
                        "start_sec": scene.start_sec,
                        "end_sec": scene.end_sec,
                        "score": score,
                        "scene_summary": scene.summary,
                        "visual_tags": scene.visual_tags,
                        "subjects": list(getattr(scene, "subjects", None) or []),
                        "scene_category": scene.scene_category,
                        "quality": scene.quality,
                    })

        candidates.sort(key=lambda x: (-x["score"], -x["quality"]))

        if not candidates and assets:
            # 关键词完全不匹配时，返回按质量排序的所有素材
            for asset in assets:
                for i, scene in enumerate(asset.scenes):
                    if scene.quality < min_quality:
                        continue
                    candidates.append({
                        "asset_id": asset.asset_id,
                        "source_path": asset.source_path,
                        "start_sec": scene.start_sec,
                        "end_sec": scene.end_sec,
                        "score": 0.0,
                        "scene_summary": scene.summary,
                        "visual_tags": scene.visual_tags,
                    "subjects": list(getattr(scene, "subjects", None) or []),
                        "subjects": list(getattr(scene, "subjects", None) or []),
                        "scene_category": scene.scene_category,
                        "quality": scene.quality,
                    })
            candidates.sort(key=lambda x: -x["quality"])

        clips = [CandidateClip(**c) for c in candidates[:top_k]]

        categories = self._categorize_clips(clips)
        total_duration = self._calc_total_duration(clips)
        estimated_output = self._estimate_output_duration(len(clips), total_duration)

        return CandidateClips(
            query=query,
            destination=destination or "",
            total_found=len(candidates),
            clips=clips,
            categories=categories,
            total_available_duration=total_duration,
            estimated_output_duration=estimated_output,
        )

    def retrieve_all(
        self,
        destination: str | None = None,
        min_quality: float = 0.0,
    ) -> CandidateClips:
        """获取某目的地所有可用素材（素材驱动模式用）

        不做关键词过滤，返回符合目的地的全部素材，
        并按类别分组，供剧本生成器决定用哪些。
        """
        logger.info(f"获取全部素材: destination={destination}")

        if destination:
            assets = self.asset_store.filter_by_destination(destination)
        else:
            assets = list(self.asset_store.list_all())

        candidates = []
        for asset in assets:
            for scene in asset.scenes:
                if scene.quality < min_quality:
                    continue
                candidates.append({
                    "asset_id": asset.asset_id,
                    "source_path": asset.source_path,
                    "start_sec": scene.start_sec,
                    "end_sec": scene.end_sec,
                    "score": scene.quality,
                    "scene_summary": scene.summary,
                    "visual_tags": scene.visual_tags,
                    "subjects": list(getattr(scene, "subjects", None) or []),
                    "scene_category": scene.scene_category,
                    "quality": scene.quality,
                })

        candidates.sort(key=lambda x: -x["quality"])

        clips = [CandidateClip(**c) for c in candidates]
        categories = self._categorize_clips(clips)
        total_duration = self._calc_total_duration(clips)
        estimated_output = self._estimate_output_duration(len(clips), total_duration)

        logger.info(
            f"共获取 {len(clips)} 个素材片段, "
            f"总时长 {total_duration:.0f}s, "
            f"估算输出 {estimated_output:.0f}s, "
            f"{len(categories)} 个类别"
        )

        return CandidateClips(
            query="all_assets",
            destination=destination or "",
            total_found=len(candidates),
            clips=clips,
            categories=categories,
            total_available_duration=total_duration,
            estimated_output_duration=estimated_output,
        )

    def _categorize_clips(self, clips: list[CandidateClip]) -> list[SceneCategory]:
        """将素材按场景类型分类

        优先使用 clip.scene_category（AI 分析结果），
        没有时回退到关键词匹配。
        """
        category_map: dict[str, SceneCategory] = {}

        for cat_key, cat_info in SCENE_CATEGORIES.items():
            category_map[cat_key] = SceneCategory(
                category=cat_key,
                display_name=cat_info["display_name"],
                count=0,
                clips=[],
            )

        uncategorized = SceneCategory(
            category="other",
            display_name="其他",
            count=0,
            clips=[],
        )

        for clip in clips:
            # 优先使用 AI 分析出的 scene_category
            if clip.scene_category and clip.scene_category != "other":
                if clip.scene_category in category_map:
                    category_map[clip.scene_category].clips.append(clip)
                    category_map[clip.scene_category].count += 1
                    continue

            # 回退：关键词匹配
            clip_text = (clip.scene_summary + " " + " ".join(clip.visual_tags)).lower()

            best_cat = None
            best_score = 0

            for cat_key, cat_info in SCENE_CATEGORIES.items():
                score = sum(
                    1 for kw in cat_info["keywords"]
                    if kw in clip_text
                )
                if score > best_score:
                    best_score = score
                    best_cat = cat_key

            if best_cat and best_score > 0:
                category_map[best_cat].clips.append(clip)
                category_map[best_cat].count += 1
            else:
                uncategorized.clips.append(clip)
                uncategorized.count += 1

        result = [cat for cat in category_map.values() if cat.count > 0]
        result.sort(key=lambda c: -c.count)

        if uncategorized.count > 0:
            result.append(uncategorized)

        return result

    @staticmethod
    def _calc_total_duration(clips: list[CandidateClip]) -> float:
        """计算素材总时长"""
        return sum(clip.end_sec - clip.start_sec for clip in clips)

    @staticmethod
    def _estimate_output_duration(num_clips: int, total_duration: float) -> float:
        """估算输出视频时长

        规则：
        - 精华率 15-25%（素材越短，精华率越低）
        - 最少 15 秒，最多 60 秒（短视频平台友好）
        - 每个镜头平均 2.5-4 秒
        """
        if num_clips == 0:
            return 0.0

        avg_shot_duration = 3.0  # 平均每镜头 3 秒
        by_count = num_clips * avg_shot_duration

        essence_ratio = 0.2  # 精华率 20%
        by_duration = total_duration * essence_ratio

        estimated = min(by_count, by_duration)
        estimated = max(15.0, min(60.0, estimated))

        return round(estimated, 1)
