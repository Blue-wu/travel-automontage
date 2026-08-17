"""多模态模型客户端 — 封装 Gemini / CLIP / Whisper 调用

遵循 OpenMontage 方法论：路径键缓存避免重复上传，结构化输出。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.common.config import CACHE_DIR, get_settings

logger = logging.getLogger(__name__)

# 旅行场景分类体系 — 用于剧本生成的内容匹配
TRAVEL_SCENE_CATEGORIES = [
    "snow_mountain",   # 雪山
    "lake",            # 湖泊
    "grassland",       # 草原
    "forest",          # 森林
    "desert",          # 沙漠
    "canyon",          # 峡谷
    "river",           # 河流/瀑布
    "sea",             # 大海/海滩
    "road",            # 公路/自驾
    "sunset",          # 日落/日出
    "starry_sky",      # 星空/银河
    "sky",             # 天空/云海
    "architecture",    # 建筑/人文
    "city",            # 城市
    "food",            # 美食
    "people",          # 人物/自拍
    "animal",          # 动物
    "flower",          # 花海/花朵
    "reflection",      # 倒影
    "aerial",          # 航拍/俯视
]


class ModelClient:
    """统一的多模态模型客户端

    优先使用 Gemini API 进行视频分析；
    如果 Gemini API 不可用（无 API key），回退到 ffprobe + 本地分析模式。
    """

    def __init__(self):
        self.settings = get_settings()
        self._gemini_file_cache: dict[str, str] = self._load_file_cache()
        self._analysis_cache: dict[str, dict] = self._load_analysis_cache()
        self._genai = None
        self._clip_model = None
        self._clip_classifier = None
        self._video_extractor = None
        self._qwen_analyzer = None
        self._multimodal_embedder = None

    # ── 缓存管理 ──────────────────────────────────────────

    def _load_file_cache(self) -> dict[str, str]:
        cache_path = CACHE_DIR / "gemini_file_cache.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        return {}

    def _save_file_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "gemini_file_cache.json").write_text(
            json.dumps(self._gemini_file_cache, indent=2, ensure_ascii=False)
        )

    def _load_analysis_cache(self) -> dict[str, dict]:
        cache_path = CACHE_DIR / "video_analysis_cache.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        return {}

    def _save_analysis_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "video_analysis_cache.json").write_text(
            json.dumps(self._analysis_cache, indent=2, ensure_ascii=False)
        )

    def _file_key(self, video_path: str) -> str:
        """基于路径+大小+修改时间生成缓存键"""
        p = Path(video_path)
        stat = p.stat()
        raw = f"{p.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Gemini 客户端初始化 ──────────────────────────────────────────

    def _get_genai(self):
        """延迟初始化 Gemini 客户端"""
        if self._genai is not None:
            return self._genai

        api_key = self.settings.gemini.api_key
        if not api_key:
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            return genai
        except ImportError:
            logger.warning("google-generativeai 未安装，Gemini 分析不可用")
            return None

    # ── Qwen-VL / 多模态向量 初始化 ──────────────────────────────────

    def _get_qwen_analyzer(self):
        """延迟初始化 Qwen-VL 分析器"""
        if self._qwen_analyzer is not None:
            return self._qwen_analyzer if self._qwen_analyzer.is_available() else None

        try:
            from tools.common.qwen_vl_analyzer import QwenVLAnalyzer
            analyzer = QwenVLAnalyzer(
                api_key=self.settings.dashscope.api_key,
                vl_model=self.settings.dashscope.vl_model,
            )
            if analyzer.is_available():
                self._qwen_analyzer = analyzer
                return analyzer
        except Exception as e:
            logger.debug(f"Qwen-VL 初始化失败: {e}")

        return None

    def _get_multimodal_embedder(self):
        """延迟初始化多模态向量器"""
        if self._multimodal_embedder is not None:
            return self._multimodal_embedder if self._multimodal_embedder.is_available() else None

        try:
            from tools.common.multimodal_embedding import MultimodalEmbedder
            embedder = MultimodalEmbedder(
                api_key=self.settings.dashscope.api_key,
                model=self.settings.dashscope.embedding_model,
                dim=self.settings.dashscope.embedding_dim,
            )
            if embedder.is_available():
                self._multimodal_embedder = embedder
                return embedder
        except Exception as e:
            logger.debug(f"多模态向量器初始化失败: {e}")

        return None

    # ── 主入口：视频场景分析 ──────────────────────────────────────────

    def analyze_video_scenes(
        self,
        video_path: str,
        destination: str = "",
        use_qwen: bool = True,
    ) -> dict[str, Any]:
        """分析视频，返回结构化场景摘要

        策略（优先级从高到低）：
        1. 缓存
        2. Qwen-VL 视频分析（有 DASHSCOPE_API_KEY，推荐）
        3. Gemini 整视频上传分析（有 API key 且视频 <20MB）
        4. Gemini 抽帧图片分析（有 API key，大视频）
        5. CLIP 零样本分类（本地免费，无需 API key）
        6. ffprobe 基础分析（最终回退）

        Returns:
            {
                "destination": "新疆-赛里木湖",
                "scenes": [...],
                "quality_score": 0.9,
                "metadata": {...}
            }
        """
        # 查缓存
        key = self._file_key(video_path)
        if key in self._analysis_cache:
            logger.debug(f"视频分析命中缓存: {Path(video_path).name}")
            return self._analysis_cache[key]

        result = None

        # 优先：Qwen-VL 视频分析
        if use_qwen and self._get_qwen_analyzer() is not None:
            logger.info(f"使用 Qwen-VL 视频分析: {Path(video_path).name}")
            result = self._qwen_analyzer.analyze_video(video_path, destination=destination)

        # 回退 1：Gemini 分析
        if result is None:
            genai = self._get_genai()
            if genai is not None:
                file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
                if file_size_mb < 20:
                    logger.info(f"使用 Gemini 整视频分析: {Path(video_path).name} ({file_size_mb:.1f}MB)")
                    result = self._analyze_with_gemini_full(video_path, genai)
                if result is None:
                    logger.info(f"使用 Gemini 抽帧分析: {Path(video_path).name}")
                    result = self._analyze_with_gemini_frames(video_path, genai)

        # 回退 2：CLIP 零样本分类
        if result is None:
            logger.info(f"使用 CLIP 零样本分类: {Path(video_path).name}")
            result = self._analyze_with_clip(video_path)

        # 最终回退
        if result is None:
            logger.info(f"使用 ffprobe 基础分析: {Path(video_path).name}")
            result = self._analyze_with_ffprobe(video_path)

        # 写入缓存
        self._analysis_cache[key] = result
        self._save_analysis_cache()

        return result

    # ── V2: 镜头切分 + 多帧 CLIP + VideoMAE ─────────────────────────

    def analyze_video_scenes_v2(
        self,
        video_path: str,
        use_video_embedding: bool = False,
    ) -> dict[str, Any]:
        """V2 视频分析：内容感知镜头切分 + 多帧 CLIP

        Args:
            video_path: 视频路径
            use_video_embedding: 是否提取 VideoMAE 视频特征（较慢，默认关闭）

        流程：
        1. PySceneDetect 内容感知镜头切分
        2. 每个镜头均匀抽 3-7 帧
        3. CLIP 多帧分类投票 + 均值池化 embedding
        4. (可选) VideoMAE 视频级特征提取
        5. 构建结构化结果
        """
        key = self._file_key(video_path) + f"_v2_{'video' if use_video_embedding else 'cliponly'}"
        if key in self._analysis_cache:
            logger.debug(f"V2 视频分析命中缓存: {Path(video_path).name}")
            return self._analysis_cache[key]

        logger.info(f"V2 视频分析 (镜头切分+多帧, video_emb={use_video_embedding}): {Path(video_path).name}")

        result = self._analyze_v2_shot_based(video_path, use_video_embedding)

        self._analysis_cache[key] = result
        self._save_analysis_cache()
        return result

    def _analyze_v2_shot_based(self, video_path: str, use_video_embedding: bool = False) -> dict[str, Any]:
        """基于镜头切分的 V2 分析实现"""
        from tools.common.shot_detector import detect_shots
        from tools.common.video_feature_extractor import aggregate_frame_embeddings

        probe = self.ffprobe(video_path)
        duration = float(probe.get("format", {}).get("duration", 0))
        if duration <= 0:
            return self._analyze_with_ffprobe(video_path)

        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        # Step 1: 镜头切分
        shots = detect_shots(video_path)
        if not shots:
            return self._analyze_with_ffprobe(video_path)

        # Step 2: 初始化 CLIP 分类器
        try:
            from tools.common.clip_classifier import CLIPZeroShotClassifier, CATEGORY_DISPLAY_NAMES
            if self._clip_classifier is None:
                self._clip_classifier = CLIPZeroShotClassifier()
        except Exception as e:
            logger.warning(f"CLIP 分类器初始化失败: {e}")
            return self._analyze_with_ffprobe(video_path)

        # Step 3: VideoMAE 视频特征提取器（可选，全局单例）
        video_extractor = None
        if use_video_embedding:
            try:
                from tools.common.video_feature_extractor import VideoFeatureExtractor
                if self._video_extractor is None:
                    self._video_extractor = VideoFeatureExtractor()
                video_extractor = self._video_extractor
            except Exception as e:
                logger.debug(f"VideoMAE 不可用: {e}")

        import tempfile
        frames_dir = Path(tempfile.mkdtemp(prefix="v2_frames_"))

        scenes = []
        all_tags = set()
        all_categories = set()
        total_quality = 0.0

        for i, shot in enumerate(shots):
            start_sec = shot["start_sec"]
            end_sec = shot["end_sec"]
            shot_dur = end_sec - start_sec

            # 抽帧数量：短镜头少抽，长镜头多抽
            if shot_dur <= 2:
                n_frames = 3
            elif shot_dur <= 5:
                n_frames = 5
            else:
                n_frames = 7

            # 均匀抽帧
            from tools.common.shot_detector import extract_frames_from_shot
            frame_paths = extract_frames_from_shot(
                video_path, start_sec, end_sec, n_frames, str(frames_dir)
            )

            if not frame_paths:
                scenes.append(self._make_fallback_scene(start_sec, end_sec, i, audio_stream))
                continue

            # 每帧 CLIP 分类 + 向量化
            clip_results = []
            frame_embeddings = []
            for fp in frame_paths:
                result = self._clip_classifier.classify_frame(fp)
                clip_results.append(result)
                # 用分类器的 open_clip 做编码（和分类同一模型空间）
                emb = self._clip_classifier.encode_image(fp)
                if emb:
                    frame_embeddings.append(emb)

            # 多帧分类投票
            category_votes: dict[str, float] = {}
            all_frame_tags = []
            for cr in clip_results:
                cat = cr.get("category", "other")
                conf = cr.get("confidence", 0.5)
                category_votes[cat] = category_votes.get(cat, 0) + conf
                all_frame_tags.extend(cr.get("top_tags", []))

            best_category = max(category_votes, key=category_votes.get) if category_votes else "other"
            best_confidence = category_votes[best_category] / len(clip_results) if clip_results else 0.5

            # 去重标签，取出现最多的
            from collections import Counter
            tag_counts = Counter(all_frame_tags)
            top_tags = [t for t, _ in tag_counts.most_common(6)]

            # CLIP 多帧聚合 embedding
            scene_embedding = None
            if frame_embeddings:
                scene_embedding = aggregate_frame_embeddings(frame_embeddings, method="mean")

            # VideoMAE 视频特征（可选）
            video_embedding = None
            if video_extractor is not None and shot_dur >= 1.5:
                try:
                    video_embedding = video_extractor.extract_shot(video_path, start_sec, end_sec)
                except Exception as e:
                    logger.debug(f"VideoMAE 提取失败 (镜头{i}): {e}")

            # 质量估算
            quality = min(0.9, 0.5 + best_confidence * 0.4)

            display_name = CATEGORY_DISPLAY_NAMES.get(best_category, "其他")

            scene = {
                "start": self._seconds_to_timecode(start_sec),
                "end": self._seconds_to_timecode(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "summary": f"{display_name}镜头",
                "visual_tags": top_tags,
                "scene_category": best_category,
                "motion_tags": [],
                "audio_tags": ["有音频"] if audio_stream else ["无音频"],
                "quality": round(quality, 2),
                "people_count": 0,
                "dominant_colors": [],
                "embedding": scene_embedding,
                "video_embedding": video_embedding,
            }
            scenes.append(scene)
            total_quality += quality
            all_tags.update(top_tags)
            all_categories.add(best_category)

        if not scenes:
            return self._analyze_with_ffprobe(video_path)

        avg_quality = total_quality / len(scenes) if scenes else 0.7

        return {
            "destination": "",
            "scenes": scenes,
            "quality_score": round(avg_quality, 2),
            "all_tags": list(all_tags),
            "categories": list(all_categories),
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "analysis_method": "v2_shot_multiframe",
            },
        }

    # ── Gemini 整视频分析 ──────────────────────────────────────────

    def _analyze_with_gemini_full(self, video_path: str, genai) -> dict[str, Any] | None:
        """使用 Gemini 整视频上传分析（质量最好，但慢且贵）"""
        # 上传文件（带缓存）
        key = self._file_key(video_path)
        if key in self._gemini_file_cache:
            file_uri = self._gemini_file_cache[key]
        else:
            try:
                file_obj = genai.upload_file(path=video_path)
                file_uri = file_obj.uri
                self._gemini_file_cache[key] = file_uri
                self._save_file_cache()
            except Exception as e:
                logger.warning(f"Gemini 视频上传失败: {e}")
                return None

        model = genai.GenerativeModel(self.settings.gemini.model)
        prompt = self._build_video_analysis_prompt()

        try:
            response = model.generate_content(
                [prompt, {"file_data": {"file_uri": file_uri}}],
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
            )
            text = response.text.strip()
            result = self._parse_json_response(text)
            if result:
                return self._normalize_analysis_result(result, video_path)
        except Exception as e:
            logger.warning(f"Gemini 整视频分析失败: {e}")

        return None

    # ── Gemini 抽帧分析（推荐：快且省） ──────────────────────────────────

    def _analyze_with_gemini_frames(self, video_path: str, genai) -> dict[str, Any] | None:
        """抽帧分析模式：先 ffprobe 分段，每段抽帧用 Gemini 图片分析

        优点：
        - 比整视频上传快 5-10 倍
        - 成本低很多
        - 可以控制分析粒度
        """
        probe = self.ffprobe(video_path)
        duration = float(probe.get("format", {}).get("duration", 0))
        if duration <= 0:
            return None

        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        # 分段：每 5-8 秒一段，太短的视频不分
        if duration <= 10:
            scene_durations = [duration]
        elif duration <= 30:
            n_scenes = max(2, int(duration / 6))
            scene_durations = [duration / n_scenes] * n_scenes
        else:
            n_scenes = max(3, min(8, int(duration / 8)))
            scene_durations = [duration / n_scenes] * n_scenes

        # 对每一段抽中间帧并分析
        frames_dir = CACHE_DIR / "analysis_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        scenes = []
        cumulative = 0.0
        total_quality = 0.0
        all_tags = set()
        all_categories = set()
        destination_guess = ""

        model = genai.GenerativeModel(self.settings.gemini.model)

        for i, sd in enumerate(scene_durations):
            start_sec = cumulative
            end_sec = min(cumulative + sd, duration)
            mid_sec = (start_sec + end_sec) / 2
            cumulative = end_sec

            # 抽帧
            frame_path = frames_dir / f"{Path(video_path).stem}_{i}.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-ss", str(mid_sec), "-i", video_path,
                        "-frames:v", "1", "-q:v", "3",
                        "-vf", "scale=640:-1",
                        str(frame_path), "-y",
                    ],
                    check=True, capture_output=True,
                )
            except Exception as e:
                logger.warning(f"抽帧失败: {e}")
                scenes.append(self._make_fallback_scene(start_sec, end_sec, i, audio_stream))
                continue

            # 用 Gemini 分析图片
            try:
                prompt = self._build_frame_analysis_prompt(i + 1, start_sec, end_sec)
                with open(frame_path, "rb") as f:
                    image_bytes = f.read()

                response = model.generate_content(
                    [
                        prompt,
                        {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
                    ],
                    generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
                )
                text = response.text.strip()
                scene_data = self._parse_json_response(text)

                if scene_data and isinstance(scene_data, dict):
                    scene = {
                        "start": self._seconds_to_timecode(start_sec),
                        "end": self._seconds_to_timecode(end_sec),
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "summary": scene_data.get("summary", f"片段 {i+1}"),
                        "visual_tags": scene_data.get("visual_tags", []),
                        "scene_category": scene_data.get("scene_category", "other"),
                        "motion_tags": scene_data.get("motion_tags", []),
                        "audio_tags": ["有音频"] if audio_stream else ["无音频"],
                        "quality": float(scene_data.get("quality", 0.7)),
                        "people_count": scene_data.get("people_count", 0),
                        "dominant_colors": scene_data.get("dominant_colors", []),
                    }
                    scenes.append(scene)
                    total_quality += scene["quality"]
                    all_tags.update(scene["visual_tags"])
                    if scene.get("scene_category"):
                        all_categories.add(scene["scene_category"])
                    if scene_data.get("destination"):
                        destination_guess = scene_data["destination"]
                else:
                    scenes.append(self._make_fallback_scene(start_sec, end_sec, i, audio_stream))

            except Exception as e:
                logger.warning(f"第 {i+1} 帧分析失败: {e}")
                scenes.append(self._make_fallback_scene(start_sec, end_sec, i, audio_stream))

            # 避免 rate limit
            time.sleep(0.5)

        if not scenes:
            return None

        avg_quality = total_quality / len(scenes) if scenes else 0.7

        return {
            "destination": destination_guess,
            "scenes": scenes,
            "quality_score": round(avg_quality, 2),
            "all_tags": list(all_tags),
            "categories": list(all_categories),
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "analysis_method": "gemini_frame",
            }
        }

    # ── CLIP 零样本分类（本地免费） ──────────────────────────────────

    def _analyze_with_clip(self, video_path: str) -> dict[str, Any] | None:
        """使用 CLIP 零样本分类分析视频（本地运行，免费）

        流程：
        1. ffprobe 获取视频信息
        2. 自动分段（每 6-8 秒一段）
        3. 每段抽中间帧
        4. CLIP 零样本分类 → 场景类别 + 视觉标签
        5. 构建结构化结果
        """
        try:
            from tools.common.clip_classifier import (
                CLIPZeroShotClassifier,
                CATEGORY_DISPLAY_NAMES,
                extract_frame,
            )
        except ImportError as e:
            logger.warning(f"CLIP 分类器导入失败: {e}")
            return None

        probe = self.ffprobe(video_path)
        duration = float(probe.get("format", {}).get("duration", 0))
        if duration <= 0:
            return None

        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        # 分段策略：CLIP 模式下每视频只抽 3 帧（开头/中间/结尾），够用且快
        if duration <= 5:
            timestamps = [duration / 2]  # 太短只抽 1 帧
        elif duration <= 15:
            timestamps = [duration * 0.3, duration * 0.7]  # 2 帧
        else:
            timestamps = [duration * 0.2, duration * 0.5, duration * 0.8]  # 3 帧

        # 初始化分类器（首次会下载模型，约 600MB）
        if self._clip_classifier is None:
            try:
                self._clip_classifier = CLIPZeroShotClassifier()
            except Exception as e:
                logger.warning(f"CLIP 分类器初始化失败: {e}")
                return None

        frames_dir = CACHE_DIR / "clip_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        scenes = []
        all_tags = set()
        all_categories = set()
        total_quality = 0.0

        n_frames = len(timestamps)
        segment_dur = duration / n_frames if n_frames > 0 else duration

        for i, mid_sec in enumerate(timestamps):
            start_sec = max(0, mid_sec - segment_dur / 2)
            end_sec = min(duration, mid_sec + segment_dur / 2)

            # 抽帧
            frame_path = frames_dir / f"{Path(video_path).stem}_{i}.jpg"
            success = extract_frame(str(video_path), mid_sec, str(frame_path))

            if not success or not frame_path.exists():
                scenes.append(self._make_fallback_scene(start_sec, end_sec, i, audio_stream))
                continue

            # CLIP 分类
            clip_result = self._clip_classifier.classify_frame(str(frame_path))

            category = clip_result.get("category", "other")
            confidence = clip_result.get("confidence", 0.5)
            top_tags = clip_result.get("top_tags", [])
            display_name = CATEGORY_DISPLAY_NAMES.get(category, "其他")

            # 质量估算：基于置信度 + 分辨率因素
            quality = min(0.9, 0.5 + confidence * 0.4)

            scene = {
                "start": self._seconds_to_timecode(start_sec),
                "end": self._seconds_to_timecode(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "summary": f"{display_name}场景",
                "visual_tags": top_tags,
                "scene_category": category,
                "motion_tags": [],
                "audio_tags": ["有音频"] if audio_stream else ["无音频"],
                "quality": round(quality, 2),
                "people_count": 0,
                "dominant_colors": [],
            }
            scenes.append(scene)
            total_quality += quality
            all_tags.update(top_tags)
            all_categories.add(category)

        if not scenes:
            return None

        avg_quality = total_quality / len(scenes) if scenes else 0.7

        return {
            "destination": "",
            "scenes": scenes,
            "quality_score": round(avg_quality, 2),
            "all_tags": list(all_tags),
            "categories": list(all_categories),
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "analysis_method": "clip_zeroshot",
            }
        }

    # ── ffprobe 基础分析（最终回退） ──────────────────────────────────

    def _analyze_with_ffprobe(self, video_path: str) -> dict[str, Any]:
        """无 API key 时的回退方案：用 ffprobe 做基础分析"""
        probe = self.ffprobe(video_path)

        duration = float(probe.get("format", {}).get("duration", 0))
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            fps = 30.0

        scene_duration = 8.0
        num_scenes = max(1, int(duration / scene_duration))
        scenes = []
        for i in range(num_scenes):
            start_sec = i * scene_duration
            end_sec = min((i + 1) * scene_duration, duration)
            scenes.append({
                "start": self._seconds_to_timecode(start_sec),
                "end": self._seconds_to_timecode(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "summary": f"片段 {i+1}",
                "visual_tags": [],
                "scene_category": "other",
                "motion_tags": [],
                "audio_tags": ["有音频"] if audio_stream else ["无音频"],
                "quality": 0.7,
                "people_count": 0,
                "dominant_colors": [],
            })

        return {
            "destination": "",
            "scenes": scenes,
            "quality_score": 0.7,
            "all_tags": [],
            "categories": [],
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "analysis_method": "ffprobe",
            }
        }

    # ── Prompt 构建 ──────────────────────────────────────────

    def _build_video_analysis_prompt(self) -> str:
        """整视频分析的 prompt"""
        categories_str = ", ".join(TRAVEL_SCENE_CATEGORIES)
        return f"""你是专业的旅行视频分析师。请仔细观看这个旅行视频，按场景分段输出结构化 JSON。

要求：
1. 按画面内容变化切分场景（不是固定时长），场景数量 3-10 个
2. 每个场景包含时间码、描述、标签、分类、质量评分
3. visual_tags 用具体、有画面感的词，不要笼统（如"雪山倒影"比"风景"好）
4. scene_category 从以下分类中选最匹配的一个：{categories_str}
5. quality 是画面质量分（0-1），考虑构图、光线、稳定性、美感
6. 识别目的地（国家-城市/景点），不确定就留空

输出 JSON 格式：
{{
  "destination": "新疆-赛里木湖",
  "scenes": [
    {{
      "start": "00:00:00",
      "end": "00:00:08.5",
      "start_sec": 0.0,
      "end_sec": 8.5,
      "summary": "航拍视角俯瞰湛蓝的赛里木湖，雪山环绕",
      "visual_tags": ["赛里木湖", "航拍", "雪山", "湛蓝湖水", "环湖公路"],
      "scene_category": "lake",
      "motion_tags": ["航拍推进", "缓慢移动"],
      "audio_tags": ["风声", "音乐"],
      "quality": 0.92,
      "people_count": 0,
      "dominant_colors": ["蓝色", "白色", "绿色"]
    }}
  ],
  "quality_score": 0.88
}}

只输出 JSON，不要其他文字，不要 markdown 代码块。"""

    def _build_frame_analysis_prompt(self, scene_num: int, start_sec: float, end_sec: float) -> str:
        """单帧分析的 prompt"""
        categories_str = ", ".join(TRAVEL_SCENE_CATEGORIES)
        return f"""你是专业的旅行摄影师。请分析这张视频截图（来自视频第 {start_sec:.1f}-{end_sec:.1f} 秒，第 {scene_num} 个场景）。

请输出 JSON：
{{
  "summary": "一句话描述画面内容，要有画面感（如'雪山脚下的草原上，牛羊在吃草'）",
  "visual_tags": ["标签1", "标签2", "标签3", "标签4", "标签5"],
  "scene_category": "最匹配的分类，从这些选一个：{categories_str}",
  "motion_tags": ["推测的镜头运动，如'静态'、'推近'、'横移'、'航拍'"],
  "quality": 0.85,
  "people_count": 0,
  "dominant_colors": ["蓝色", "绿色"],
  "destination": "能识别的话写景点名，不确定就空字符串"
}}

要求：
- visual_tags 要具体，有画面感，5-8 个词
- scene_category 必须从给定列表选最接近的
- quality 是画面美感分 0-1（构图、光线、色彩）
- 只输出 JSON，不要其他文字"""

    # ── 工具方法 ──────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict | None:
        """解析模型返回的 JSON"""
        text = text.strip()
        # 清理 markdown 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 找第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"JSON 解析失败: {text[:100]}")
            return None

    def _normalize_analysis_result(self, result: dict, video_path: str) -> dict:
        """标准化整视频分析结果的格式（与抽帧模式对齐）"""
        scenes = result.get("scenes", [])
        all_tags = set()
        categories = set()

        for s in scenes:
            tags = s.get("visual_tags", [])
            all_tags.update(tags)
            if "scene_category" in s:
                categories.add(s["scene_category"])
            # 补齐抽帧模式有的字段
            if "people_count" not in s:
                s["people_count"] = 0
            if "dominant_colors" not in s:
                s["dominant_colors"] = []
            if "scene_category" not in s:
                s["scene_category"] = "other"

        probe = self.ffprobe(video_path)
        duration = float(probe.get("format", {}).get("duration", 0))
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        return {
            "destination": result.get("destination", ""),
            "scenes": scenes,
            "quality_score": float(result.get("quality_score", 0.7)),
            "all_tags": list(all_tags),
            "categories": list(categories),
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": float(video_stream.get("r_frame_rate", "30/1").split("/")[0]) if "/" in video_stream.get("r_frame_rate", "30/1") else 30.0,
                "analysis_method": "gemini_full",
            }
        }

    def _make_fallback_scene(
        self, start_sec: float, end_sec: float, index: int, audio_stream
    ) -> dict:
        """生成一个 fallback 场景数据"""
        return {
            "start": self._seconds_to_timecode(start_sec),
            "end": self._seconds_to_timecode(end_sec),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "summary": f"片段 {index+1}",
            "visual_tags": [],
            "scene_category": "other",
            "motion_tags": [],
            "audio_tags": ["有音频"] if audio_stream else ["无音频"],
            "quality": 0.6,
            "people_count": 0,
            "dominant_colors": [],
        }

    # ── 向量化（多模态向量优先，CLIP 回退）──────────────────────────

    def embed_text(self, text: str) -> list[float] | None:
        """将文本转为向量

        优先用多模态向量模型（与视频同一空间），回退到 CLIP。
        """
        # 优先：多模态向量（和视频向量同一空间）
        embedder = self._get_multimodal_embedder()
        if embedder is not None:
            vec = embedder.embed_text(text)
            if vec is not None:
                return vec

        # 回退：CLIP
        try:
            if self._clip_classifier is None:
                from tools.common.clip_classifier import CLIPZeroShotClassifier
                self._clip_classifier = CLIPZeroShotClassifier()
            return self._clip_classifier.encode_text(text)
        except Exception:
            pass

        clip_encoder = self._get_clip()
        if clip_encoder is not None:
            return clip_encoder.encode_text(text)
        return None

    def embed_image(self, image_path: str) -> list[float] | None:
        """将图像转为向量"""
        # 优先：多模态向量
        embedder = self._get_multimodal_embedder()
        if embedder is not None:
            vec = embedder.embed_image(image_path)
            if vec is not None:
                return vec

        # 回退：CLIP
        try:
            if self._clip_classifier is None:
                from tools.common.clip_classifier import CLIPZeroShotClassifier
                self._clip_classifier = CLIPZeroShotClassifier()
            return self._clip_classifier.encode_image(image_path)
        except Exception:
            pass

        clip_encoder = self._get_clip()
        if clip_encoder is not None:
            return clip_encoder.encode_image(image_path)
        return None

    def embed_video_clip(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> list[float] | None:
        """视频片段 → 向量（多模态向量模型）

        这是方案 C 的核心：直接把视频片段编码为向量，
        和文本向量在同一空间，可以做精准的"文本→视频"检索。
        """
        embedder = self._get_multimodal_embedder()
        if embedder is not None:
            return embedder.embed_video(video_path, start_sec, end_sec)
        return None

    def embed_video_frame(self, video_path: str, timestamp_sec: float) -> list[float] | None:
        """提取视频帧并转为向量（兼容旧接口）"""
        # 优先：用多模态向量对视频片段编码（取 timestamp 前后 2 秒）
        embedder = self._get_multimodal_embedder()
        if embedder is not None:
            start = max(0, timestamp_sec - 2)
            vec = embedder.embed_video(video_path, start, timestamp_sec + 2)
            if vec is not None:
                return vec

        # 回退：抽帧 + CLIP
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame_path = f.name

        try:
            from tools.common.clip_classifier import extract_frame
            if extract_frame(video_path, timestamp_sec, frame_path):
                return self.embed_image(frame_path)
        finally:
            try:
                Path(frame_path).unlink(missing_ok=True)
            except Exception:
                pass

        clip_encoder = self._get_clip()
        if clip_encoder is not None:
            return clip_encoder.encode_video_frame(video_path, timestamp_sec)
        return None

    def _get_clip(self):
        """延迟初始化 CLIP 模型"""
        if self._clip_model is not None:
            return self._clip_model

        try:
            import clip
            import torch
            model, preprocess = clip.load(self.settings.clip.model_name, device=self.settings.clip.device)
            self._clip_model = CLIPEncoderWrapper(model, preprocess, self.settings.clip.device)
            return self._clip_model
        except ImportError:
            return None

    # ── ffprobe 工具 ──────────────────────────────────────────

    @staticmethod
    def ffprobe(video_path: str) -> dict:
        """运行 ffprobe 获取视频信息"""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    video_path,
                ],
                capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

    @staticmethod
    def _seconds_to_timecode(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"


class CLIPEncoderWrapper:
    """CLIP 编码器封装"""

    def __init__(self, model, preprocess, device: str):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def encode_text(self, text: str) -> list[float]:
        import clip
        import torch
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().tolist()

    def encode_image(self, image_path: str) -> list[float]:
        from PIL import Image
        import torch
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().tolist()

    def encode_video_frame(self, video_path: str, timestamp_sec: float) -> list[float]:
        """提取视频指定时间戳的帧并编码"""
        frame_path = str(CACHE_DIR / f"frame_{hash(video_path)}_{int(timestamp_sec)}.jpg")
        subprocess.run(
            [
                "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", frame_path, "-y"
            ],
            check=True, capture_output=True,
        )
        return self.encode_image(frame_path)
