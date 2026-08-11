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
from tools.common.models import CandidateClip, CandidateClips

logger = logging.getLogger(__name__)


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

            if asset and scene_index < len(asset.scenes):
                scene = asset.scenes[scene_index]
                quality = scene.quality
                source_path = asset.source_path
                start_sec = scene.start_sec
                end_sec = scene.end_sec
                visual_tags = scene.visual_tags

            if quality < min_quality:
                continue

            # 余弦相似度
            emb_vec = np.array(embedding, dtype=np.float32)
            sim = float(np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec) + 1e-8
            ))

            scored.append({
                "asset_id": asset_id,
                "source_path": source_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "score": sim,
                "scene_summary": summary,
                "visual_tags": visual_tags,
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
                        "quality": scene.quality,
                    })

        candidates.sort(key=lambda x: -x["score"])
        clips = [CandidateClip(**c) for c in candidates[:top_k]]

        return CandidateClips(
            query=query,
            destination=destination or "",
            total_found=len(candidates),
            clips=clips,
        )
