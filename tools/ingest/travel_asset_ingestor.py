"""素材入库模块 — 把"杂乱素材"变成"可被语义检索的资产库"

流程：扫描目录 → 转码 → 上传模型API（带路径缓存）→ 逐段分析 → 向量化 → 写入素材库
参考 Montai 的做法：转码为模型可处理格式，上传到 Gemini File API（路径键缓存避免重复上传），
逐段分析每个视频生成场景摘要。
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Iterator

from tools.common.asset_store import AssetStore
from tools.common.config import CACHE_DIR, get_settings
from tools.common.model_client import ModelClient
from tools.common.models import Asset, AssetMetadata, Scene

logger = logging.getLogger(__name__)

# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}


class TravelAssetIngestor:
    """旅行素材入库器

    扫描素材目录 → 转码 → 上传模型API → 逐段分析 → 写入素材库

    用法：
        ingestor = TravelAssetIngestor()
        ingestor.ingest("~/Travel/Fuji2024")
    """

    def __init__(
        self,
        asset_store: AssetStore | None = None,
        model_client: ModelClient | None = None,
    ):
        self.asset_store = asset_store or AssetStore()
        self.model_client = model_client or ModelClient()
        self.settings = get_settings()

    def ingest(
        self,
        footage_dir: str,
        skip_existing: bool = True,
        skip_transcode: bool = False,
        destination: str | None = None,
        use_v2: bool = True,
        hybrid_mode: bool = True,
    ) -> list[Asset]:
        """入库整个目录的素材

        Args:
            footage_dir: 素材目录路径
            skip_existing: 是否跳过已入库的素材（基于路径去重）
            skip_transcode: 是否跳过转码步骤
            destination: 手动指定目的地
            use_v2: 是否使用 V2 分析（镜头切分 + 多帧 + VideoMAE）
            hybrid_mode: 混合模式（本地 CLIP + qwen 精选增强，省 token）

        Returns:
            入库成功的 Asset 列表
        """
        footage_path = Path(footage_dir).expanduser()
        if not footage_path.exists():
            raise FileNotFoundError(f"素材目录不存在: {footage_path}")

        video_list = list(self._scan_videos(footage_path))
        total = len(video_list)
        ingested = []
        for idx, video_path in enumerate(video_list, 1):
            try:
                if skip_existing:
                    existing = self.asset_store.get_by_path(str(video_path))
                    if existing is not None:
                        logger.info(f"[{idx}/{total}] 跳过已入库: {video_path.name}")
                        ingested.append(existing)
                        continue

                logger.info(f"[{idx}/{total}] 入库: {video_path.name}")
                asset = self._ingest_single(
                    video_path,
                    skip_transcode=skip_transcode,
                    override_destination=destination,
                    use_v2=use_v2,
                    hybrid_mode=hybrid_mode,
                )
                if asset:
                    self.asset_store.save(asset)
                    ingested.append(asset)
                    logger.info(
                        f"[{idx}/{total}] 入库成功: {video_path.name} "
                        f"({len(asset.scenes)} 场景)"
                    )
                else:
                    logger.warning(f"[{idx}/{total}] 入库失败: {video_path.name}")
            except Exception as e:
                logger.error(f"[{idx}/{total}] 入库异常 {video_path.name}: {e}")

        logger.info(f"入库完成: {len(ingested)}/{total} 成功")
        return ingested

    def _ingest_single(
        self,
        video_path: Path,
        skip_transcode: bool = False,
        override_destination: str | None = None,
        use_v2: bool = True,
        hybrid_mode: bool = True,
        qwen_quality_threshold: float = 0.75,
    ) -> Asset | None:
        """入库单个视频文件

        Args:
            hybrid_mode: 混合模式（默认开启）
                - 本地 CLIP 做场景切分 + 分类 + 向量化（免费）
                - 只对精彩场景用 qwen 生成描述（省 token）
            qwen_quality_threshold: 精彩场景质量阈值，超过才调用 qwen 增强描述
        """
        asset_id = self._generate_asset_id(str(video_path))

        # Step 1: 转码为统一格式
        if skip_transcode:
            normalized_path = video_path
        else:
            normalized_path = self._transcode(video_path)

        dest_hint = override_destination or ""

        if hybrid_mode:
            # ── 混合策略：本地 CLIP 分析 + qwen 精选增强 ──
            analysis = self._hybrid_analyze(
                str(normalized_path), dest_hint, qwen_quality_threshold
            )
        elif use_v2:
            # V2 纯本地分析
            analysis = self.model_client.analyze_video_scenes_v2(
                str(normalized_path), use_video_embedding=False
            )
            analysis["destination"] = dest_hint
        else:
            # 旧策略：Qwen-VL 优先
            analysis = self.model_client.analyze_video_scenes(
                str(normalized_path), destination=dest_hint, use_qwen=True
            )

        # Step 3: 构建场景列表 + 向量化
        scenes = self._build_scenes_with_embeddings(
            analysis, str(normalized_path), str(video_path)
        )

        # Step 4: 构建 Asset
        metadata = self._build_metadata(analysis, str(video_path))

        dest = override_destination or analysis.get("destination", "")

        return Asset(
            asset_id=asset_id,
            source_path=str(video_path),
            normalized_path=str(normalized_path),
            destination=dest,
            scenes=scenes,
            metadata=metadata,
        )

    def _hybrid_analyze(
        self,
        video_path: str,
        destination: str,
        quality_threshold: float = 0.75,   # 保留形参兼容旧调用，新路径不再使用
    ) -> dict:
        """混合分析策略（已重构）

        旧流程的问题：主路径是 CLIP 零样本分类 + 类别→关键词查表，
        一段视频被压成「1 个枚举 + 一组类别同义词」。这一步有损且不可逆，
        画面里的具体物件（没化完的浮冰、牧民的摩托车）在入库时就丢了。
        后果是检索粒度过粗、文案抓不出错位手法 —— 同一个根因的两个症状。

        新流程：
          主路径  VLM 整段分析 → 分段 + 结构化字段（1 次调用/视频，便宜）
          回退    PySceneDetect + CLIP 分类（仅 VLM 不可用时）
          之后    本地 CLIP 向量（免费，检索用）+ 本地画质指标（覆盖 quality）

        注意：本次不改动向量来源，仍走与旧路径一致的本地 CLIP，
        避免把「打标改造」和「向量空间改造」两个变化混在一起而无法归因。
        """
        import logging
        log = logging.getLogger(__name__)

        qwen = self.model_client._get_qwen_analyzer()
        if qwen is not None:
            log.info(f"结构化打标 [VLM 整段]: {Path(video_path).name}")
            try:
                analysis = qwen.analyze_video(video_path, destination=destination)
            except Exception as e:
                log.warning(f"  VLM 分析异常，回退本地 CLIP: {e}")
                analysis = None

            if analysis and analysis.get("scenes"):
                analysis["destination"] = destination or analysis.get("destination", "")
                analysis["analysis_method"] = "vlm_structured"
                n_subj = sum(len(sc.get("subjects") or []) for sc in analysis["scenes"])
                log.info(
                    f"  {len(analysis['scenes'])} 个场景，"
                    f"{n_subj} 个具体物件（错位手法抓手）"
                )
                self._attach_local_signals(analysis, video_path)
                return analysis
            log.warning("  VLM 未返回可用场景，回退本地 CLIP")
        else:
            log.info("  VLM 不可用（无 DASHSCOPE_API_KEY），走本地 CLIP 回退路径")

        # ── 回退：镜头切分 + CLIP 零样本分类（信息量有限，仅保证流程不断）──
        analysis = self.model_client.analyze_video_scenes_v2(
            video_path, use_video_embedding=False
        )
        analysis["destination"] = destination
        analysis["analysis_method"] = "clip_fallback"
        self._attach_local_signals(analysis, video_path)
        return analysis

    def _attach_local_signals(self, analysis: dict, video_path: str) -> None:
        """为每个场景补：本地 CLIP 向量（若缺）+ 画质指标，并重算 quality

        画质指标替代原先的关键词加分。原实现用 high_value_keywords 命中数推
        quality，而那份词表含"震撼""绝美" —— 正是文案规范禁用的套话词，
        等于系统在奖励套话；且 q = 0.68 + score 把大量场景抬到 0.68-0.92，
        区分度基本丧失。
        """
        import logging
        import tempfile
        log = logging.getLogger(__name__)

        scenes = analysis.get("scenes") or []
        if not scenes:
            return

        try:
            from tools.common.frame_quality import aggregate_metrics, fuse_quality
            from tools.common.shot_detector import extract_frames_from_shot
        except Exception as e:
            log.debug(f"画质模块不可用，跳过: {e}")
            return

        clf = None
        try:
            from tools.common.clip_classifier import CLIPZeroShotClassifier
            if getattr(self.model_client, "_clip_classifier", None) is None:
                self.model_client._clip_classifier = CLIPZeroShotClassifier()
            clf = self.model_client._clip_classifier
        except Exception as e:
            log.debug(f"CLIP 编码器不可用，向量将由下游回退处理: {e}")

        frames_dir = Path(tempfile.mkdtemp(prefix="signals_"))
        for sc in scenes:
            # 在可用区间内抽帧，避开起幅落幅
            a = float(sc.get("usable_start_sec") or sc.get("start_sec", 0.0))
            b = float(sc.get("usable_end_sec") or sc.get("end_sec", a + 1.0))
            if b <= a:
                b = a + 1.0
            try:
                frames = extract_frames_from_shot(
                    video_path, a, b, 4, str(frames_dir)
                )
            except Exception:
                continue
            if not frames:
                continue

            # 画质指标 → 融合 quality
            m = aggregate_metrics(frames)
            if m:
                sc["sharpness"] = m.get("sharpness", 0.0)
                sc["brightness"] = m.get("brightness", 0.0)
                sc["exposure_ok"] = m.get("exposure_ok", True)
            sc["quality"] = fuse_quality(
                sc.get("quality", 0.5), m, sc.get("defects") or []
            )

            # 本地 CLIP 向量（保持与旧路径同一空间）
            if not sc.get("embedding") and clf is not None:
                try:
                    from tools.common.video_feature_extractor import (
                        aggregate_frame_embeddings,
                    )
                    embs = [e for e in (clf.encode_image(f) for f in frames) if e]
                    if embs:
                        sc["embedding"] = aggregate_frame_embeddings(embs, method="mean")
                except Exception as e:
                    log.debug(f"向量化失败（非致命）: {e}")

    def _qwen_describe_frame(
        self,
        qwen_analyzer,
        frame_path: str,
        destination: str,
        category_hint: str,
    ) -> dict | None:
        """用 qwen 对单张关键帧生成描述（省 token 模式）

        相比整视频上传，单张图片只消耗 ~500-1000 token
        """
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = qwen_analyzer.api_key

        dest_hint = f"拍摄于{destination}。" if destination else ""
        cat_hint = f"CLIP初步分类为{category_hint}。" if category_hint else ""

        prompt = f"""请分析这张旅行视频截图。{dest_hint}{cat_hint}

只返回JSON，不要其他文字：
{{
  "summary": "20-40字描述画面内容，要有画面感和情绪",
  "visual_tags": ["具体标签1", "具体标签2", "具体标签3", "具体标签4", "具体标签5"],
  "quality": 0.85,
  "dominant_colors": ["颜色1", "颜色2"]
}}

quality评分标准：0.9+顶级震撼，0.8优秀，0.7良好，0.6可用，0.5一般。"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{frame_path}"},
                    {"text": prompt},
                ],
            }
        ]

        try:
            response = MultiModalConversation.call(
                model=qwen_analyzer.vl_model,
                messages=messages,
                result_format="message",
            )

            if response.status_code != 200:
                return None

            text = ""
            for choice in response.output.choices:
                content = choice.message.content
                if isinstance(content, str):
                    text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text += item["text"]

            # 解析 JSON
            import json
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return None

        except Exception:
            return None

    def _scan_videos(self, directory: Path) -> Iterator[Path]:
        """扫描目录中的视频文件"""
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path

    def _generate_asset_id(self, video_path: str) -> str:
        """基于文件路径 + 大小 + 修改时间生成唯一 ID"""
        p = Path(video_path)
        stat = p.stat()
        raw = f"{p.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _transcode(self, video_path: Path) -> Path:
        """转码为统一格式（H.264, 保持原分辨率, 30fps）

        如果已经是标准格式则直接返回原路径。
        """
        cache_dir = CACHE_DIR / "normalized"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_path = cache_dir / f"{video_path.stem}_normalized.mp4"

        # 如果缓存已存在且比源文件新，直接用缓存
        if output_path.exists() and output_path.stat().st_mtime > video_path.stat().st_mtime:
            return output_path

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-r", "30",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            "-y", str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"转码失败 {video_path}: {e.stderr[:200]}")
            # 转码失败则用原文件
            return video_path

    # ── 场景分类规范化映射 ──────────────────────────────────────────
    # 将模型可能返回的各种分类名映射到标准分类
    _CATEGORY_NORMALIZE_MAP = {
        # 航拍相关
        "aerial": "aerial",
        "aerial_view": "aerial",
        "drone": "aerial",
        "overhead": "aerial",
        "俯视": "aerial",
        "航拍": "aerial",
        # 雪山
        "snow_mountain": "snow_mountain",
        "snow": "snow_mountain",
        "mountain": "snow_mountain",
        "glacier": "snow_mountain",
        "peak": "snow_mountain",
        "雪山": "snow_mountain",
        "山峰": "snow_mountain",
        # 草原
        "grassland": "grassland",
        "meadow": "grassland",
        "prairie": "grassland",
        "pasture": "grassland",
        "草原": "grassland",
        "牧场": "grassland",
        # 湖泊
        "lake": "lake",
        "pond": "lake",
        "天池": "lake",
        "湖泊": "lake",
        # 河流
        "river": "river",
        "stream": "river",
        "waterfall": "river",
        "valley": "river",
        "河流": "river",
        "溪流": "river",
        "瀑布": "river",
        # 森林
        "forest": "forest",
        "woods": "forest",
        "trees": "forest",
        "森林": "forest",
        "树林": "forest",
        # 峡谷
        "canyon": "canyon",
        "cliff": "canyon",
        "gorge": "canyon",
        "yadan": "canyon",
        "峡谷": "canyon",
        # 沙漠
        "desert": "desert",
        "gobi": "desert",
        "dune": "desert",
        "沙漠": "desert",
        "戈壁": "desert",
        # 大海
        "sea": "sea",
        "ocean": "sea",
        "beach": "sea",
        "coast": "sea",
        "大海": "sea",
        "海滩": "sea",
        # 公路
        "road": "road",
        "highway": "road",
        "transport": "road",
        "drive": "road",
        "公路": "road",
        "自驾": "road",
        "道路": "road",
        # 日出日落
        "sunset": "sunset",
        "sunrise": "sunset",
        "dawn": "sunset",
        "dusk": "sunset",
        "日落": "sunset",
        "日出": "sunset",
        # 星空
        "starry_sky": "starry_sky",
        "stars": "starry_sky",
        "galaxy": "starry_sky",
        "night": "starry_sky",
        "星空": "starry_sky",
        "银河": "starry_sky",
        "夜景": "starry_sky",
        # 天空云海
        "sky": "sky",
        "clouds": "sky",
        "cloud": "sky",
        "fog": "sky",
        "天空": "sky",
        "云海": "sky",
        "云彩": "sky",
        "雾气": "sky",
        # 建筑
        "architecture": "architecture",
        "building": "architecture",
        "temple": "architecture",
        "village": "architecture",
        "建筑": "architecture",
        "古迹": "architecture",
        "寺庙": "architecture",
        "村落": "architecture",
        # 城市
        "city": "city",
        "town": "city",
        "street": "city",
        "城市": "city",
        "城镇": "city",
        # 花海
        "flower": "flower",
        "flowers": "flower",
        "blossom": "flower",
        "花海": "flower",
        "花朵": "flower",
        # 倒影
        "reflection": "reflection",
        "mirror": "reflection",
        "倒影": "reflection",
        # 人物
        "people": "people",
        "person": "people",
        "portrait": "people",
        "人物": "people",
        "人像": "people",
        "自拍": "people",
        "landscape": "sky",  # landscape太笼统，默认归为天空/风景
    }

    @classmethod
    def _normalize_category(cls, cat: str) -> str:
        """将任意分类名规范化为标准分类"""
        if not cat:
            return "other"
        # 先直接查
        if cat in cls._CATEGORY_NORMALIZE_MAP:
            return cls._CATEGORY_NORMALIZE_MAP[cat]
        # 大小写不敏感
        lower = cat.lower().strip()
        if lower in cls._CATEGORY_NORMALIZE_MAP:
            return cls._CATEGORY_NORMALIZE_MAP[lower]
        # 关键词包含匹配
        for key, val in cls._CATEGORY_NORMALIZE_MAP.items():
            if key in lower:
                return val
        # 标准分类列表
        std_cats = {
            "snow_mountain", "grassland", "lake", "river", "forest", "canyon",
            "desert", "sea", "road", "aerial", "sunset", "starry_sky", "sky",
            "architecture", "city", "flower", "reflection", "people", "food",
            "animal", "other",
        }
        if cat in std_cats:
            return cat
        return "other"

    def _build_scenes_with_embeddings(
        self,
        analysis: dict,
        normalized_path: str,
        source_path: str,
    ) -> list[Scene]:
        """从分析结果构建场景列表，并对每个场景做视频向量化

        向量化策略：
        1. 优先用多模态向量模型对视频片段编码（文本-视频同一空间）
        2. 回退到 CLIP 抽帧向量化
        """
        scenes_data = analysis.get("scenes", [])
        scenes = []

        for i, s in enumerate(scenes_data):
            # 规范化分类
            raw_cat = s.get("scene_category", "other")
            normalized_cat = self._normalize_category(raw_cat)

            # quality 已由 _attach_local_signals 用「VLM 判断 + 本地画质指标」算过，
            # 不再做关键词加分（原 _enhance_quality 会奖励"震撼""绝美"这类
            # 文案规范明令禁用的词，且把分布抬到 0.68-0.92 丧失区分度）
            start_sec = float(s.get("start_sec", 0))
            end_sec = float(s.get("end_sec", 10))

            scene = Scene(
                start=s.get("start", "00:00:00"),
                end=s.get("end", "00:00:10"),
                start_sec=start_sec,
                end_sec=end_sec,
                summary=s.get("summary", ""),
                visual_tags=s.get("visual_tags", []),
                scene_category=normalized_cat,
                motion_tags=s.get("motion_tags", []),
                audio_tags=s.get("audio_tags", []),
                quality=float(s.get("quality", 0.7)),
                people_count=int(s.get("people_count", 0)),
                dominant_colors=s.get("dominant_colors", []),
                # ── VLM 结构化字段 ──
                subjects=s.get("subjects", []),
                shot_scale=s.get("shot_scale", ""),
                camera_motion=s.get("camera_motion", ""),
                motion_class=s.get("motion_class", ""),
                time_of_day=s.get("time_of_day", "unknown"),
                weather=s.get("weather", "unknown"),
                mood=s.get("mood", "neutral"),
                has_person=bool(s.get("has_person", False)),
                has_speech=bool(s.get("has_speech", False)),
                ambient_sound=s.get("ambient_sound", []),
                defects=s.get("defects", []),
                # ── 本地画质指标 ──
                sharpness=float(s.get("sharpness", 0.0)),
                brightness=float(s.get("brightness", 0.0)),
                exposure_ok=bool(s.get("exposure_ok", True)),
                usable_start_sec=float(s.get("usable_start_sec") or start_sec),
                usable_end_sec=float(s.get("usable_end_sec") or end_sec),
            )

            # 如果分析结果已包含 embedding，直接用
            if s.get("embedding"):
                scene.embedding = s["embedding"]
                if s.get("video_embedding"):
                    scene.video_embedding = s["video_embedding"]
            else:
                # 对视频片段做向量化
                try:
                    logger.info(
                        f"  场景 {i+1}/{len(scenes_data)} 向量化: "
                        f"{scene.start_sec:.1f}s-{scene.end_sec:.1f}s"
                    )
                    embedding = self.model_client.embed_video_clip(
                        normalized_path,
                        start_sec=scene.start_sec,
                        end_sec=scene.end_sec,
                    )
                    if embedding:
                        scene.embedding = embedding
                    else:
                        # 回退：抽中间帧 CLIP
                        mid_sec = (scene.start_sec + scene.end_sec) / 2
                        embedding = self.model_client.embed_video_frame(
                            normalized_path, mid_sec
                        )
                        if embedding:
                            scene.embedding = embedding
                except Exception as e:
                    logger.debug(f"向量化失败（非致命）: {e}")

            scenes.append(scene)

        return scenes

    def _build_metadata(self, analysis: dict, source_path: str) -> AssetMetadata:
        """构建素材元数据"""
        meta = analysis.get("metadata", {})
        probe = self.model_client.ffprobe(source_path)

        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = float(probe.get("format", {}).get("duration", meta.get("duration", 0)))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        return AssetMetadata(
            duration=duration,
            has_audio=audio_stream is not None,
            quality_score=float(analysis.get("quality_score", 0.7)),
            resolution=f"{width}x{height}",
            fps=fps,
            tags=analysis.get("tags", []),
        )
