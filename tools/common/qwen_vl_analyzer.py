"""Qwen 视频分析模块 — 基于阿里云 DashScope

用 qwen3.8-max 等支持视频理解的模型分析视频内容，
生成结构化的场景描述、分类、标签和精彩度评分。

核心优势：
- 原生视频理解（不需要手动抽帧）
- 中文理解强
- 秒级时间戳定位
- 支持长视频（1小时+）
- 通用大模型，能力更强
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QwenVLAnalyzer:
    """Qwen 视频分析器

    用法：
        analyzer = QwenVLAnalyzer(api_key="sk-xxx")
        result = analyzer.analyze_video("video.mp4", destination="新疆")
    """

    def __init__(self, api_key: str = "", vl_model: str = "qwen3.8-max"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.vl_model = vl_model
        self._available: bool | None = None

    def is_available(self) -> bool:
        """检查是否可用（有 API Key）"""
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key)
        if self._available:
            logger.info("Qwen-VL 分析器已就绪")
        return self._available

    def analyze_video(
        self,
        video_path: str,
        destination: str = "",
        max_size_mb: float = 100.0,
    ) -> dict[str, Any] | None:
        """分析整个视频，返回结构化场景信息

        Args:
            video_path: 视频文件路径
            destination: 目的地名称
            max_size_mb: 视频最大允许大小（MB），超过会压缩

        Returns:
            分析结果，与 ModelClient.analyze_video_scenes 格式一致
        """
        if not self.is_available():
            logger.warning("Qwen-VL 不可用（无 API Key）")
            return None

        # 检查视频大小，超过限制先压缩
        actual_path = video_path
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            logger.info(f"视频过大 ({file_size_mb:.1f}MB)，压缩到 {max_size_mb}MB 以内...")
            actual_path = self._compress_video(video_path, max_size_mb)
            if actual_path is None:
                logger.warning("视频压缩失败，跳过")
                return None

        try:
            result = self._call_qwen_vl(actual_path, destination)
        finally:
            # 清理临时压缩文件
            if actual_path != video_path and os.path.exists(actual_path):
                os.unlink(actual_path)

        return result

    def _call_qwen_vl(self, video_path: str, destination: str) -> dict[str, Any] | None:
        """调用 qwen3.8-max API 分析视频（使用 DashScope SDK，支持本地文件上传）"""
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = self.api_key

        # 构建分析 prompt
        prompt = self._build_analysis_prompt(destination)

        # 使用 DashScope SDK 原生格式，SDK 会自动上传本地视频文件
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": f"file://{video_path}"},
                    {"text": prompt},
                ],
            }
        ]

        logger.info(f"调用 {self.vl_model} 分析: {Path(video_path).name}")

        try:
            response = MultiModalConversation.call(
                model=self.vl_model,
                messages=messages,
                result_format="message",
            )

            if response.status_code != 200:
                logger.error(f"{self.vl_model} API 失败: {response.code} - {response.message}")
                return None

            # 提取返回文本
            text = ""
            for choice in response.output.choices:
                content = choice.message.content
                # content 可能是 str 或 list[dict]
                if isinstance(content, str):
                    text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text += item["text"]
                        elif isinstance(item, str):
                            text += item

            return self._parse_result(text, video_path, destination)

        except Exception as e:
            logger.error(f"{self.vl_model} 调用异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_analysis_prompt(self, destination: str) -> str:
        """构建视频分析 prompt"""
        dest_hint = f"这是在{destination}拍摄的旅行视频。" if destination else ""

        return f"""请仔细分析这段旅行视频。{dest_hint}

请先完整观看视频，理解每个镜头的内容，然后按场景分段输出结构化JSON。

## 场景分类体系（scene_category必须从以下列表中选择最匹配的一个）：
- snow_mountain: 雪山、冰川、山峰
- grassland: 草原、牧场、草甸、牛羊群
- lake: 湖泊、天池、水塘、湖面倒影
- river: 河流、溪流、瀑布、河谷
- forest: 森林、树林、针叶林、白桦林
- canyon: 峡谷、悬崖、峭壁、雅丹地貌
- desert: 沙漠、戈壁、沙丘
- sea: 大海、海滩、海岸线
- road: 公路、自驾、道路、开车、沿途风光
- aerial: 航拍、无人机、俯视、上帝视角（当主要镜头是航拍时用这个）
- sunset: 日落、日出、黄昏、晨光
- starry_sky: 星空、银河、夜景
- sky: 天空、云海、云彩、雾气
- architecture: 建筑、古迹、寺庙、村落、人文建筑
- city: 城市、街道、城镇
- flower: 花海、花朵、野花、花丛
- reflection: 水面倒影、镜子反射
- people: 人物、人像、自拍、人群
- food: 美食、餐饮、食物
- animal: 动物、牛羊、马、野生动物
- other: 其他无法归类的场景

注意：航拍优先看内容，如果是航拍草原就用grassland，如果是纯航拍视角（看不出具体内容）才用aerial。

## 质量评分标准（quality 0-1）：
- 0.9+：顶级素材，构图完美，光线绝佳，画面震撼，有强烈视觉冲击力（如：日照金山、星空银河、绝美倒影、震撼航拍全景）
- 0.8-0.89：优秀素材，构图好，光线佳，画面美丽（如：草原牛羊、湖泊风光、雪山远景）
- 0.7-0.79：良好素材，画面清晰，内容不错（如：普通公路风景、一般近景）
- 0.6-0.69：可用素材，画面一般但能用（如：普通特写、轻微晃动）
- 0.5-0.59：一般素材，有瑕疵（如：构图一般、光线不足）
- <0.5：差素材，不建议使用（如：严重模糊、过曝、内容空）

## 输出JSON格式：
{{
  "destination": "{destination}",
  "scenes": [
    {{
      "start_sec": 0.0,
      "end_sec": 5.0,
      "summary": "详细描述画面内容，要有画面感，20-50字，包含氛围和情绪（如：航拍视角俯瞰翠绿的草原，牛羊散落其间，远山如黛，宁静辽阔）",
      "visual_tags": ["标签1", "标签2", "标签3", "标签4", "标签5"],
      "scene_category": "snow_mountain",
      "motion_tags": ["航拍推进"],
      "audio_tags": ["有背景音乐"],
      "quality": 0.85,
      "people_count": 0,
      "dominant_colors": ["绿色", "蓝色"]
    }}
  ],
  "quality_score": 0.85,
  "all_tags": ["草原", "雪山", "航拍"]
}}

## 要求：
1. 按画面内容变化自然分段，每段时长3-15秒，总场景数2-10个
2. start_sec/end_sec 精确到0.1秒
3. visual_tags 5-8个，用中文具体名词，描述画面里的关键元素（如"赛里木湖"比"湖泊"好，"天山雪峰"比"雪山"好）
4. scene_category 必须从上面的分类列表选最匹配的一个
5. motion_tags 1-3个，描述镜头运动方式（推、拉、平移、环绕、上升、下降、固定、航拍、摇镜等）
6. quality 严格按上面的评分标准打分
7. 只返回JSON，不要其他文字，不要markdown代码块"""

    def _parse_result(
        self, text: str, video_path: str, destination: str
    ) -> dict[str, Any] | None:
        """解析 Qwen-VL 返回的 JSON 结果"""
        # 清理可能的 markdown 包裹
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.error(f"JSON 解析失败: {text[:200]}")
                    return None
            else:
                logger.error(f"无法提取 JSON: {text[:200]}")
                return None

        # 补充元数据
        probe = self._ffprobe(video_path)
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(probe.get("format", {}).get("duration", 0))

        # 确保 scenes 有 timecode 格式
        for scene in result.get("scenes", []):
            if "start" not in scene:
                scene["start"] = self._seconds_to_timecode(scene.get("start_sec", 0))
            if "end" not in scene:
                scene["end"] = self._seconds_to_timecode(scene.get("end_sec", 0))

        result["destination"] = destination or result.get("destination", "")
        result["metadata"] = {
            "duration": duration,
            "has_audio": audio_stream is not None,
            "resolution": f"{width}x{height}",
            "fps": self._parse_fps(video_stream.get("r_frame_rate", "30/1")),
            "analysis_method": "qwen_vl",
        }

        return result

    def _compress_video(self, video_path: str, target_size_mb: float) -> str | None:
        """压缩视频到目标大小以内"""
        probe = self._ffprobe(video_path)
        duration = float(probe.get("format", {}).get("duration", 0))
        if duration <= 0:
            return None

        # 计算目标码率
        target_bitrate = int(target_size_mb * 8 * 1024 * 1024 / duration * 0.9)  # 留 10% 余量
        target_bitrate = min(target_bitrate, 4_000_000)  # 最高 4Mbps

        output_path = str(Path(tempfile.mktemp(suffix=".mp4", prefix="qwen_compress_")))

        cmd = [
            "ffmpeg", "-i", video_path,
            "-c:v", "libx264",
            "-b:v", str(target_bitrate),
            "-maxrate", str(int(target_bitrate * 1.5)),
            "-bufsize", str(target_bitrate * 2),
            "-preset", "fast",
            "-vf", "scale=-2:720",  # 降到 720p
            "-c:a", "aac",
            "-b:a", "64k",
            "-y", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            compressed_size = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"压缩完成: {compressed_size:.1f}MB")
            return output_path
        except Exception as e:
            logger.warning(f"压缩失败: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None

    def _ffprobe(self, video_path: str) -> dict:
        """获取视频元数据"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path],
                capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)
        except Exception:
            return {}

    def _parse_fps(self, fps_str: str) -> float:
        try:
            return eval(fps_str) if "/" in fps_str else float(fps_str)
        except Exception:
            return 30.0

    def _seconds_to_timecode(self, sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
