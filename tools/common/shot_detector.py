"""镜头切分模块 — 基于 PySceneDetect 的内容感知镜头检测

用内容感知算法（ContentDetector）检测镜头切换点，
替代固定时长分段，让每个场景对应一个真正的"镜头"。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def detect_shots(
    video_path: str,
    threshold: float = 18.0,
    min_scene_len: float = 1.5,
    max_scene_len: float = 10.0,
) -> list[dict[str, Any]]:
    """检测视频中的镜头切换点

    Args:
        video_path: 视频文件路径
        threshold: 内容变化阈值（越小越敏感，默认 18.0）
        min_scene_len: 最短镜头时长（秒），避免碎镜头
        max_scene_len: 最长镜头时长（秒），超长镜头会被强制切分

    Returns:
        镜头列表，每个包含 start_sec, end_sec
        [
            {"start_sec": 0.0, "end_sec": 3.5},
            {"start_sec": 3.5, "end_sec": 8.2},
            ...
        ]
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError:
        logger.warning("PySceneDetect 未安装，回退到固定时长分段")
        return _fallback_fixed_segments(video_path, max_scene_len)

    try:
        video = open_video(video_path)
    except Exception as e:
        logger.warning(f"打开视频失败: {e}")
        return _fallback_fixed_segments(video_path, max_scene_len)

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_len * video.frame_rate),
        )
    )

    try:
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
    except Exception as e:
        logger.warning(f"镜头检测失败: {e}")
        return _fallback_fixed_segments(video_path, max_scene_len)

    if not scene_list:
        return _fallback_fixed_segments(video_path, max_scene_len)

    shots = []
    for start, end in scene_list:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        if duration <= 0:
            continue

        if duration > max_scene_len:
            # 超长镜头强制切分
            n_sub = max(1, int(duration / max_scene_len))
            sub_dur = duration / n_sub
            for i in range(n_sub):
                sub_start = start_sec + i * sub_dur
                sub_end = min(start_sec + (i + 1) * sub_dur, end_sec)
                if sub_end - sub_start >= min_scene_len * 0.5:
                    shots.append({"start_sec": sub_start, "end_sec": sub_end})
        else:
            shots.append({"start_sec": start_sec, "end_sec": end_sec})

    if not shots:
        return _fallback_fixed_segments(video_path, max_scene_len)

    logger.debug(f"镜头检测完成: {Path(video_path).name} → {len(shots)} 个镜头")
    return shots


def _fallback_fixed_segments(video_path: str, segment_len: float) -> list[dict[str, Any]]:
    """固定时长分段回退方案"""
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0

    if duration <= 0:
        return [{"start_sec": 0.0, "end_sec": 5.0}]

    shots = []
    n_segments = max(1, int(duration / segment_len))
    actual_len = duration / n_segments
    for i in range(n_segments):
        start = i * actual_len
        end = min((i + 1) * actual_len, duration)
        shots.append({"start_sec": start, "end_sec": end})

    return shots


def extract_frames_from_shot(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = 5,
    output_dir: str | None = None,
) -> list[str]:
    """从一个镜头中均匀抽取多帧

    Args:
        video_path: 视频路径
        start_sec: 镜头开始时间
        end_sec: 镜头结束时间
        n_frames: 抽取帧数
        output_dir: 输出目录，None 则用临时目录

    Returns:
        抽取的帧文件路径列表
    """
    import subprocess
    import tempfile

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp())

    duration = max(0.1, end_sec - start_sec)
    if n_frames <= 1:
        timestamps = [start_sec + duration / 2]
    else:
        step = duration / (n_frames + 1)
        timestamps = [start_sec + step * (i + 1) for i in range(n_frames)]

    frame_paths = []
    video_stem = Path(video_path).stem

    for i, ts in enumerate(timestamps):
        frame_path = out_dir / f"{video_stem}_s{int(start_sec)}_f{i}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-ss", f"{ts:.3f}", "-i", video_path,
                    "-frames:v", "1", "-q:v", "3",
                    "-vf", "scale=512:-1",
                    str(frame_path), "-y",
                ],
                check=True, capture_output=True,
            )
            if frame_path.exists():
                frame_paths.append(str(frame_path))
        except Exception as e:
            logger.debug(f"抽帧失败 ({ts:.1f}s): {e}")

    return frame_paths
