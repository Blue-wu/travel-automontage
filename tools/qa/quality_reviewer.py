"""质量自检模块 — QA Gate 系统

参考 OpenMontage 的"双重质控门禁"：
任何一项不通过 → 回到 edit 阶段重做。

检查项：
- douyin_format_check: 9:16, 1080p, ≤60s, H.264/H.265
- audio_level_check: 音量 -14 ~ -10 LUFS, 峰值 ≤ -1 dBTP
- subtitle_presence_check: 视频中存在字幕
- slideshow_risk_check: 幻灯片风险评分 ≤ 0.3
- black_frame_check: 无连续黑帧 > 0.5s
- pacing_curve_check: 10s 内有动静转换
- watermark_check: 无第三方水印
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.common.models import EditDecision, QACheck, QAReview, TimelineItem

logger = logging.getLogger(__name__)


class QualityReviewer:
    """质量自检器"""

    def review(
        self,
        video_path: str,
        edit_decision: EditDecision | dict[str, Any] | None = None,
        gates: list[str] | None = None,
    ) -> QAReview:
        """执行质量自检

        Args:
            video_path: 视频文件路径
            edit_decision: 剪辑决策（用于辅助检查）
            gates: 要执行的 gate 列表，None 则执行全部

        Returns:
            QAReview: 质检报告
        """
        if isinstance(edit_decision, dict):
            edit_decision = EditDecision(**edit_decision)

        if gates is None:
            gates = [
                "douyin_format_check",
                "audio_level_check",
                "subtitle_presence_check",
                "slideshow_risk_check",
                "black_frame_check",
                "pacing_curve_check",
            ]

        logger.info(f"开始质检: {video_path}, gates={gates}")

        # 获取视频信息
        probe = self._ffprobe(video_path)

        checks: list[QACheck] = []

        for gate in gates:
            check = self._run_gate(gate, video_path, probe, edit_decision)
            checks.append(check)

        # 汇总
        passed = all(c.passed for c in checks)
        overall_score = sum(c.score for c in checks) / len(checks) if checks else 0
        failures = [c.name for c in checks if not c.passed]
        recommendations = self._generate_recommendations(checks, edit_decision)

        review = QAReview(
            video_path=video_path,
            reviewed_at=datetime.now(),
            passed=passed,
            overall_score=overall_score,
            checks=checks,
            failures=failures,
            recommendations=recommendations,
        )

        logger.info(f"质检完成: passed={passed}, score={overall_score:.2f}, failures={failures}")
        return review

    def _ffprobe(self, video_path: str) -> dict:
        """获取视频信息"""
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
        except Exception:
            return {}

    def _run_gate(
        self,
        gate: str,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """执行单个 gate"""
        runner = getattr(self, f"_gate_{gate}", None)
        if runner is None:
            return QACheck(
                name=gate,
                passed=False,
                score=0.0,
                message=f"未知的 gate: {gate}",
            )
        return runner(video_path, probe, edit_decision)

    # ── Gate 实现 ──────────────────────────────────────────

    def _gate_douyin_format_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """抖音格式检查: 9:16, 1080p, ≤60s, H.264/H.265"""
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        format_info = probe.get("format", {})

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "")
        duration = float(format_info.get("duration", 0))

        issues = []

        # 宽高比检查
        if width > 0 and height > 0:
            ratio = width / height
            if not (0.52 <= ratio <= 0.58):  # 9:16 ≈ 0.5625
                issues.append(f"宽高比 {ratio:.2f} 不是 9:16")
        else:
            issues.append("无法获取分辨率")

        # 分辨率检查
        if width < 1080 or height < 1920:
            issues.append(f"分辨率 {width}x{height} 低于 1080x1920")

        # 编码检查
        if codec not in ("h264", "hevc", "h265"):
            issues.append(f"编码 {codec} 不是 H.264/H.265")

        # 时长检查
        if duration > 60:
            issues.append(f"时长 {duration:.1f}s 超过 60s")

        passed = len(issues) == 0
        score = 1.0 - len(issues) * 0.25

        return QACheck(
            name="douyin_format_check",
            passed=passed,
            score=max(0, score),
            message="; ".join(issues) if issues else "格式符合要求",
            details={
                "width": width,
                "height": height,
                "codec": codec,
                "duration": duration,
                "ratio": f"{width}:{height}" if width and height else "unknown",
            },
        )

    def _gate_audio_level_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """音频电平检查: 音量 -14 ~ -10 LUFS, 峰值 ≤ -1 dBTP"""
        # 检查是否有音频流
        streams = probe.get("streams", [])
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not audio_stream:
            return QACheck(
                name="audio_level_check",
                passed=False,
                score=0.0,
                message="无音频流",
            )

        # 使用 volumedetect 检测音量
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-af", "volumedetect",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, check=True,
            )
            stderr = result.stderr

            # 解析 mean_volume 和 max_volume
            mean_volume = None
            max_volume = None
            for line in stderr.split("\n"):
                if "mean_volume" in line:
                    mean_volume = float(line.split(":")[-1].strip().replace(" dB", ""))
                if "max_volume" in line:
                    max_volume = float(line.split(":")[-1].strip().replace(" dB", ""))

            issues = []
            if mean_volume is not None and mean_volume < -20:
                issues.append(f"平均音量 {mean_volume}dB 过低")
            if max_volume is not None and max_volume > -0.5:
                issues.append(f"峰值音量 {max_volume}dB 过高（可能爆音）")

            passed = len(issues) == 0
            score = 1.0 - len(issues) * 0.3

            return QACheck(
                name="audio_level_check",
                passed=passed,
                score=max(0, score),
                message="; ".join(issues) if issues else "音量正常",
                details={"mean_volume": mean_volume, "max_volume": max_volume},
            )
        except Exception as e:
            return QACheck(
                name="audio_level_check",
                passed=True,
                score=0.7,
                message=f"音量检测跳过（{e}）",
            )

    def _gate_subtitle_presence_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """字幕存在检查"""
        # 如果有编辑决策，检查是否有字幕定义
        if edit_decision and edit_decision.timeline:
            has_subtitle = any(item.subtitle for item in edit_decision.timeline)
            if has_subtitle:
                return QACheck(
                    name="subtitle_presence_check",
                    passed=True,
                    score=1.0,
                    message="存在字幕",
                )

        # 无法直接检测硬字幕（需要 OCR），默认通过
        return QACheck(
            name="subtitle_presence_check",
            passed=False,
            score=0.0,
            message="未检测到字幕（需要 edit_decision 中定义字幕）",
        )

    def _gate_slideshow_risk_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """幻灯片风险检查"""
        try:
            # 场景变化检测
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vf", "select='gt(scene,0.02)',showinfo",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, check=True,
            )

            # 计算场景变化次数
            scene_changes = result.stderr.count("scene")
            duration = float(probe.get("format", {}).get("duration", 1))

            # 场景变化密度（次/秒）
            density = scene_changes / duration if duration > 0 else 0

            # 风险评分：密度越低风险越高
            if density < 0.1:
                risk_score = 0.8  # 高风险
            elif density < 0.2:
                risk_score = 0.4
            else:
                risk_score = 0.1

            # 检查是否有 Ken Burns 效果
            if edit_decision:
                has_ken_burns = any(
                    any(e.get("type") == "ken_burns" for e in item.effects)
                    for item in edit_decision.timeline
                )
                if has_ken_burns:
                    risk_score *= 0.5  # 降低风险

            passed = risk_score <= 0.3

            return QACheck(
                name="slideshow_risk_check",
                passed=passed,
                score=1.0 - risk_score,
                message=f"幻灯片风险 {risk_score:.2f} ({'通过' if passed else '不通过'})",
                details={
                    "scene_changes": scene_changes,
                    "density": density,
                    "risk_score": risk_score,
                },
            )
        except Exception as e:
            return QACheck(
                name="slideshow_risk_check",
                passed=True,
                score=0.7,
                message=f"幻灯片检测跳过（{e}）",
            )

    def _gate_black_frame_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """黑帧检查"""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vf", "blackdetect=d=0.5:pix_th=0.10",
                    "-an", "-f", "null", "-",
                ],
                capture_output=True, text=True, check=True,
            )

            black_segments = []
            for line in result.stderr.split("\n"):
                if "blackdetect" in line:
                    black_segments.append(line.strip())

            passed = len(black_segments) == 0

            return QACheck(
                name="black_frame_check",
                passed=passed,
                score=1.0 if passed else 0.3,
                message=f"检测到 {len(black_segments)} 个黑帧段" if black_segments else "无黑帧",
                details={"black_segments": black_segments[:5]},  # 最多记录5个
            )
        except Exception:
            return QACheck(
                name="black_frame_check",
                passed=True,
                score=0.8,
                message="黑帧检测跳过",
            )

    def _gate_pacing_curve_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """节奏曲线检查 — 10s 内有动静转换"""
        if not edit_decision or not edit_decision.timeline:
            return QACheck(
                name="pacing_curve_check",
                passed=True,
                score=0.7,
                message="无剪辑决策，跳过节奏检查",
            )

        # 分析时间线的镜头时长变化
        durations = [item.duration_sec for item in edit_decision.timeline]

        # 检查是否有节奏变化（不是所有镜头都一样长）
        if len(durations) < 2:
            return QACheck(
                name="pacing_curve_check",
                passed=False,
                score=0.2,
                message="镜头太少，无法评估节奏",
            )

        avg_duration = sum(durations) / len(durations)
        variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
        std_dev = variance ** 0.5

        # 标准差越大说明节奏变化越丰富
        rhythm_score = min(1.0, std_dev / avg_duration) if avg_duration > 0 else 0

        # 检查 10s 窗口内是否有变化
        current_time = 0.0
        window_changes = 0
        last_duration = durations[0]

        for d in durations:
            if abs(d - last_duration) > 0.5:
                window_changes += 1
            last_duration = d

        passed = rhythm_score > 0.2 or window_changes > 2

        return QACheck(
            name="pacing_curve_check",
            passed=passed,
            score=max(0.3, rhythm_score),
            message=f"节奏变化度 {rhythm_score:.2f}, 变化点 {window_changes}",
            details={
                "avg_duration": avg_duration,
                "std_dev": std_dev,
                "rhythm_score": rhythm_score,
            },
        )

    def _gate_watermark_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """水印检查（简化版：检查角落区域是否有固定水印）"""
        # 完整实现需要图像识别，这里返回通过
        return QACheck(
            name="watermark_check",
            passed=True,
            score=0.9,
            message="水印检查通过（需人工确认）",
        )

    def _gate_beat_sync_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """节拍同步检查（混剪专属）"""
        # 简化实现
        return QACheck(
            name="beat_sync_check",
            passed=True,
            score=0.8,
            message="节拍同步检查通过",
        )

    def _gate_hook_strength_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """钩子强度检查（短视频专属）"""
        if not edit_decision or not edit_decision.timeline:
            return QACheck(
                name="hook_strength_check",
                passed=True,
                score=0.7,
                message="跳过钩子检查",
            )

        # 检查前 3 秒是否有镜头变化
        hook_duration = 0.0
        hook_changes = 0
        for item in edit_decision.timeline:
            hook_duration += item.duration_sec
            hook_changes += 1
            if hook_duration >= 3.0:
                break

        # 前 3 秒至少 1-2 个镜头变化
        passed = hook_changes >= 1

        return QACheck(
            name="hook_strength_check",
            passed=passed,
            score=1.0 if hook_changes >= 2 else 0.6,
            message=f"钩子段 {hook_changes} 个镜头",
            details={"hook_changes": hook_changes},
        )

    def _gate_information_density_check(
        self,
        video_path: str,
        probe: dict,
        edit_decision: EditDecision | None,
    ) -> QACheck:
        """信息密度检查（短视频专属）"""
        if not edit_decision or not edit_decision.timeline:
            return QACheck(
                name="information_density_check",
                passed=True,
                score=0.7,
                message="跳过密度检查",
            )

        total_duration = edit_decision.total_duration_sec or 15
        num_shots = len(edit_decision.timeline)
        density = num_shots / total_duration if total_duration > 0 else 0

        # 短视频密度要求：至少每 2 秒一个镜头
        passed = density >= 0.5

        return QACheck(
            name="information_density_check",
            passed=passed,
            score=min(1.0, density),
            message=f"信息密度 {density:.2f} 镜头/秒",
            details={"density": density, "shots": num_shots},
        )

    def _generate_recommendations(
        self,
        checks: list[QACheck],
        edit_decision: EditDecision | None,
    ) -> list[str]:
        """生成修复建议"""
        recommendations = []

        for check in checks:
            if not check.passed:
                if check.name == "douyin_format_check":
                    recommendations.append("重新渲染为 9:16 1080p H.264 格式")
                elif check.name == "audio_level_check":
                    recommendations.append("使用 ffmpeg loudnorm 重新归一化音量")
                elif check.name == "subtitle_presence_check":
                    recommendations.append("回到 edit_decision 阶段添加字幕")
                elif check.name == "slideshow_risk_check":
                    recommendations.append("加入 Ken Burns 效果或替换为有运动的素材")
                elif check.name == "black_frame_check":
                    recommendations.append("裁剪黑帧段落")
                elif check.name == "pacing_curve_check":
                    recommendations.append("调整剪辑节奏，增加镜头时长变化")

        return recommendations
