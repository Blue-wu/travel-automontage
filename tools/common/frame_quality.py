"""画质指标 — 纯本地计算，替代原先「关键词命中数推质量分」的做法

原实现的问题（travel_asset_ingestor._enhance_quality）：
用 high_value_keywords 命中数推 quality，而那份词表里包含"震撼""绝美"
——正是 skills/travel-copywriting.md 明令禁用的套话词，等于系统在奖励套话。
且公式 q = 0.68 + score 把大量场景抬到 0.68-0.92，quality 区分度基本丧失。

这里改用可比较的物理量。依赖缺失时优雅降级（返回空 dict，
调用方直接采用 VLM 给的 quality）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 经验归一尺度。严格做法是全库分位数归一（见 docs/asset-pipeline-redesign.md §5.5），
# 绝对阈值在不同相机/天气下不可比。此处先用固定尺度，后续可标定。
_SHARPNESS_SCALE = 300.0
_CLIP_FRAC_LIMIT = 0.05      # 高光/暗部溢出像素占比上限


def frame_metrics(frame_path: str) -> dict[str, Any]:
    """单帧画质指标

    Returns:
        {"sharpness": 0-1, "brightness": 0-1, "exposure_ok": bool}
        依赖缺失或读取失败时返回 {}
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {}

    try:
        g = np.asarray(Image.open(frame_path).convert("L"), dtype=np.float32)
    except Exception as e:
        logger.debug(f"读取帧失败 {frame_path}: {e}")
        return {}

    if g.ndim != 2 or min(g.shape) < 3:
        return {}

    # Laplacian 方差测锐度（越大越清晰，虚焦画面接近 0）
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4.0 * g[1:-1, 1:-1])
    sharpness = min(1.0, float(lap.var()) / _SHARPNESS_SCALE)

    brightness = float(g.mean()) / 255.0
    over = float((g > 250).mean())
    under = float((g < 5).mean())
    exposure_ok = over < _CLIP_FRAC_LIMIT and under < _CLIP_FRAC_LIMIT

    return {
        "sharpness": round(sharpness, 3),
        "brightness": round(brightness, 3),
        "exposure_ok": exposure_ok,
        "overexposed_frac": round(over, 4),
        "underexposed_frac": round(under, 4),
    }


def aggregate_metrics(frame_paths: list[str]) -> dict[str, Any]:
    """多帧取均值；exposure_ok 取「全部帧都合格」"""
    ms = [m for m in (frame_metrics(p) for p in frame_paths) if m]
    if not ms:
        return {}
    n = len(ms)
    return {
        "sharpness": round(sum(m["sharpness"] for m in ms) / n, 3),
        "brightness": round(sum(m["brightness"] for m in ms) / n, 3),
        "exposure_ok": all(m["exposure_ok"] for m in ms),
    }


def fuse_quality(vlm_quality: float, metrics: dict[str, Any],
                 defects: list[str] | None = None) -> float:
    """把 VLM 的主观可用性判断和本地画质指标融合

    VLM 权重最高（它真的看过画面、懂内容价值），
    画质指标负责否掉「内容好但拍废了」的素材。
    """
    q = float(vlm_quality or 0.0)
    if not metrics:
        base = q
    else:
        base = (
            0.60 * q
            + 0.25 * metrics.get("sharpness", 0.5)
            + 0.15 * (1.0 if metrics.get("exposure_ok", True) else 0.2)
        )
    if defects:
        base *= 0.4
    return round(max(0.05, min(0.98, base)), 3)
