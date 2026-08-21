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
        """构建视频分析 prompt

        关键设计：输出的不只是一个分类枚举，而是**具体物件 + 结构化属性**。
        旧版只输出 scene_category + 一组类别同义词标签，信息在入库时就被压扁，
        导致检索粒度过粗、文案抓不出错位手法（"湖泊"抓不出，"没化完的浮冰"才能）。
        """
        from tools.common.vocab import (
            CAMERA_MOTION_CN, SCENE_CATEGORY_CN, SHOT_SCALE_CN, TIME_OF_DAY_CN,
        )

        dest_hint = f"这是在{destination}拍摄的旅行视频。" if destination else ""
        cats = "\n".join(f"- {k}: {v}" for k, v in SCENE_CATEGORY_CN.items())
        scales = "\n".join(f"- {k}: {v}" for k, v in SHOT_SCALE_CN.items())
        motions = "\n".join(f"- {k}: {v}" for k, v in CAMERA_MOTION_CN.items())
        tods = "、".join(f"{k}({v})" for k, v in TIME_OF_DAY_CN.items())

        return f"""请仔细分析这段旅行视频。{dest_hint}

先完整观看，理解每个镜头的内容和运镜，再按场景分段输出结构化 JSON。

## 最重要的一件事：subjects 必须是画面里真实存在的具体物件

这个字段决定了后续文案能不能写出有意思的东西，**不要填类别的同义词**。

✗ 错误（这些是类别同义词，等于没有信息）：
   ["湖泊","湛蓝","倒影","雪山"]   ["草原","绿色","广阔"]
✓ 正确（具体、可指认、有辨识度的东西）：
   ["没化完的浮冰","岸边碎石","远处的雪线","一只落单的水鸟"]
   ["散开的羊群","牧民的摩托车","被压倒的草","铁丝网"]

判据：**这个词能不能让人在画面里指出来？** 不能就换掉。
"湛蓝"指不出来，"浮冰"指得出来。

## 场景分类（scene_category，选最匹配的一个；这只是索引，不要用它代替 subjects）
{cats}
航拍优先看内容：航拍草原就填 grassland，只有看不出具体内容的纯航拍视角才填 aerial。

## 景别（shot_scale，按画面占比严格判断）
{scales}

## 运镜（camera_motion，只描述【相机】怎么动，不是画面里的东西怎么动）
{motions}

## 时段光线（time_of_day，按光线色温和方向判断）
{tods}

## 可用区间（usable_start_sec / usable_end_sec）
必须**去掉起幅和落幅** —— 开头镜头还没稳、结尾开始甩向别处的部分要排除。
这两个值是剪辑真正会用的入点出点，请严格判断，这直接决定成片是否毛糙。

## 质量评分（quality 0-1，要能横向比较，别都给 0.8）
- 0.9+   顶级：构图完美、光线绝佳、有强烈视觉冲击（日照金山、星空银河、绝美倒影）
- 0.8-89 优秀：构图好、光线佳
- 0.7-79 良好：画面清晰、内容不错
- 0.6-69 可用：一般但能用
- 0.5-59 有瑕疵：构图一般或光线不足
- <0.5   差：严重模糊、过曝、内容空

## 输出 JSON
{{{{
  "destination": "{destination}",
  "scenes": [
    {{{{
      "start_sec": 0.0,
      "end_sec": 5.0,
      "usable_start_sec": 0.8,
      "usable_end_sec": 4.6,
      "summary": "20-50字，描述画面里有什么，要有画面感和氛围（如：航拍俯瞰翠绿草原，羊群散落其间，远山如黛）",
      "subjects": ["具体物件1", "具体物件2", "具体物件3"],
      "visual_tags": ["标签1", "标签2", "标签3"],
      "scene_category": "grassland",
      "shot_scale": "ELS",
      "camera_motion": "aerial",
      "time_of_day": "golden_hour",
      "weather": "晴",
      "mood": "epic",
      "has_person": false,
      "has_speech": false,
      "ambient_sound": ["风声"],
      "quality": 0.85,
      "defects": [],
      "people_count": 0,
      "dominant_colors": ["绿色", "蓝色"]
    }}}}
  ],
  "quality_score": 0.85,
  "all_tags": ["草原", "雪山", "航拍"]
}}}}

## 要求
1. 按画面内容和运镜的变化自然分段，每段 3-15 秒，总场景数 2-10 个
2. start_sec / end_sec / usable_* 精确到 0.1 秒
3. subjects 3-6 个，严格遵守上面「能不能指出来」的判据
4. visual_tags 3-8 个中文具体名词（可与 subjects 重叠）
5. 完全不可用的段落（严重糊、大幅甩镜、误拍地面）也要输出，quality 给低分并在 defects 说明
6. 只返回 JSON，不要其他文字，不要 markdown 代码块"""

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

        # 规整到受控词表 + 补齐字段，防止脏值进库
        from tools.common.vocab import (
            CAMERA_MOTION, MOOD, SCENE_CATEGORY, SHOT_SCALE, TIME_OF_DAY,
            motion_class, normalize_enum,
        )
        for scene in result.get("scenes", []):
            if "start" not in scene:
                scene["start"] = self._seconds_to_timecode(scene.get("start_sec", 0))
            if "end" not in scene:
                scene["end"] = self._seconds_to_timecode(scene.get("end_sec", 0))

            scene["scene_category"] = normalize_enum(
                scene.get("scene_category", ""), SCENE_CATEGORY, "other")
            scene["shot_scale"] = normalize_enum(
                scene.get("shot_scale", ""), SHOT_SCALE, "")
            scene["camera_motion"] = normalize_enum(
                scene.get("camera_motion", ""), CAMERA_MOTION, "")
            scene["time_of_day"] = normalize_enum(
                scene.get("time_of_day", ""), TIME_OF_DAY, "unknown")
            scene["mood"] = normalize_enum(scene.get("mood", ""), MOOD, "neutral")
            scene["motion_class"] = (
                motion_class(scene["camera_motion"]) if scene["camera_motion"] else "")

            # 可用区间兜底：模型没给或给反了，退回整段
            s0, s1 = scene.get("start_sec", 0.0), scene.get("end_sec", 0.0)
            u0 = float(scene.get("usable_start_sec") or s0)
            u1 = float(scene.get("usable_end_sec") or s1)
            if not (s0 <= u0 < u1 <= s1 + 0.01):
                u0, u1 = s0, s1
            scene["usable_start_sec"], scene["usable_end_sec"] = u0, u1

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
