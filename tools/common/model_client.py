"""多模态模型客户端 — 封装 Gemini / CLIP / Whisper 调用

遵循 OpenMontage 方法论：路径键缓存避免重复上传，结构化输出。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.common.config import CACHE_DIR, get_settings


class ModelClient:
    """统一的多模态模型客户端

    优先使用 Gemini API 进行视频分析；
    如果 Gemini API 不可用（无 API key），回退到 ffprobe + 本地分析模式。
    """

    def __init__(self):
        self.settings = get_settings()
        self._gemini_file_cache: dict[str, str] = self._load_file_cache()
        self._genai = None
        self._clip_model = None

    # ── Gemini File API ──────────────────────────────────────────

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

    def _file_key(self, video_path: str) -> str:
        """基于路径+大小+修改时间生成缓存键"""
        p = Path(video_path)
        stat = p.stat()
        raw = f"{p.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()

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
            return None

    def analyze_video_scenes(self, video_path: str) -> dict[str, Any]:
        """分析视频，返回结构化场景摘要

        Returns:
            {
                "destination": "日本-富士山",
                "scenes": [...],
                "quality_score": 0.9
            }
        """
        genai = self._get_genai()
        if genai is not None:
            return self._analyze_with_gemini(video_path, genai)
        else:
            return self._analyze_with_ffprobe(video_path)

    def _analyze_with_gemini(self, video_path: str, genai) -> dict[str, Any]:
        """使用 Gemini API 分析视频"""
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
            except Exception:
                # 上传失败，回退到 ffprobe
                return self._analyze_with_ffprobe(video_path)

        model = genai.GenerativeModel(self.settings.gemini.model)

        prompt = """分析这个旅行视频，按场景分段输出 JSON：
{
  "destination": "识别的目的地（国家-城市/景点名）",
  "scenes": [
    {
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "start_sec": 0.0,
      "end_sec": 5.0,
      "summary": "场景描述",
      "visual_tags": ["标签1", "标签2"],
      "motion_tags": ["静态", "手持跟拍"],
      "audio_tags": ["人声", "风噪"],
      "quality": 0.9
    }
  ],
  "quality_score": 0.9
}
只输出 JSON，不要其他文字。"""

        try:
            response = model.generate_content([prompt, {"file_data": {"file_uri": file_uri}}])
            text = response.text.strip()
            # 清理可能的 markdown 包裹
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return self._analyze_with_ffprobe(video_path)

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
        fps = eval(fps_str) if "/" in fps_str else float(fps_str)

        # 将视频分成 10 秒一段的默认场景
        scene_duration = 10.0
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
                "motion_tags": [],
                "audio_tags": ["有音频"] if audio_stream else ["无音频"],
                "quality": 0.7,
            })

        return {
            "destination": "",
            "scenes": scenes,
            "quality_score": 0.7,
            "metadata": {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "resolution": f"{width}x{height}",
                "fps": fps,
            }
        }

    # ── CLIP 向量化 ──────────────────────────────────────────

    def embed_text(self, text: str) -> list[float] | None:
        """将文本转为 CLIP 向量"""
        clip_encoder = self._get_clip()
        if clip_encoder is None:
            return None
        return clip_encoder.encode_text(text)

    def embed_image(self, image_path: str) -> list[float] | None:
        """将图像转为 CLIP 向量"""
        clip_encoder = self._get_clip()
        if clip_encoder is None:
            return None
        return clip_encoder.encode_image(image_path)

    def embed_video_frame(self, video_path: str, timestamp_sec: float) -> list[float] | None:
        """提取视频帧并转为向量"""
        clip_encoder = self._get_clip()
        if clip_encoder is None:
            return None
        return clip_encoder.encode_video_frame(video_path, timestamp_sec)

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
