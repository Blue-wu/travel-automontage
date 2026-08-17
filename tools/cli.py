"""CLI 入口 — travel-automontage 命令行工具

用法：
    tam ingest --footage-dir ~/Travel/Fuji2024
    tam trends --niche 旅行 --destination 富士山
    tam retrieve --query "富士山日出" --destination 富士山
    tam run travel-vlog --destination 富士山 --duration 30
    tam qa --video output/final.mp4
    tam list  # 列出素材库
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from tools.common.config import OUTPUT_DIR, PIPELINE_DEFS_DIR
from tools.pipeline_runner import PipelineRunner

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def cli(verbose: bool):
    """Travel AutoMontage — 旅行赛道自动剪辑系统"""
    setup_logging(verbose)


@cli.command()
@click.option("--footage-dir", "-d", required=True, help="素材目录路径")
@click.option("--destination", "-dest", default=None, help="手动指定目的地")
@click.option("--skip-transcode", is_flag=True, default=False, help="跳过转码步骤")
@click.option("--skip-existing", is_flag=True, default=True, help="跳过已入库素材")
def ingest(footage_dir: str, destination: str | None, skip_transcode: bool, skip_existing: bool):
    """素材入库 — 转码 + AI 分析 + 向量化"""
    from tools.ingest.travel_asset_ingestor import TravelAssetIngestor

    console.print(f"[cyan]开始入库: {footage_dir}[/]")
    if destination:
        console.print(f"[cyan]目的地: {destination}[/]")
    ingestor = TravelAssetIngestor()
    assets = ingestor.ingest(
        footage_dir,
        skip_existing=skip_existing,
        skip_transcode=skip_transcode,
        destination=destination,
    )

    table = Table(title="入库结果")
    table.add_column("文件", style="cyan")
    table.add_column("目的地", style="green")
    table.add_column("场景数", justify="right")
    table.add_column("时长", justify="right")
    table.add_column("质量", justify="right")

    for asset in assets:
        table.add_row(
            Path(asset.source_path).name,
            asset.destination or "-",
            str(len(asset.scenes)),
            f"{asset.metadata.duration:.1f}s",
            f"{asset.metadata.quality_score:.2f}",
        )

    console.print(table)
    console.print(f"\n[green]✓ 入库完成: {len(assets)} 个素材[/]")


@cli.command()
@click.option("--niche", "-n", default="旅行", help="赛道")
@click.option("--destination", "-d", default=None, help="目的地过滤")
@click.option("--focus", "-f", default=None, help="关注方向（hooks/pacing/music）")
@click.option("--output", "-o", default=None, help="输出文件路径")
def trends(niche: str, destination: str | None, focus: str | None, output: str | None):
    """抖音趋势分析"""
    from tools.trends.douyin_trend_analyzer import DouyinTrendAnalyzer

    console.print(f"[cyan]趋势分析: niche={niche}, destination={destination}[/]")
    analyzer = DouyinTrendAnalyzer()

    # 加载人工样本
    samples_dir = Path("data/samples/douyin_viral")
    if samples_dir.exists():
        analyzer.load_manual_samples(samples_dir)

    report = analyzer.analyze_travel_trends(niche=niche, destination=destination, focus=focus)

    # 打印报告
    console.print(f"\n[bold green]趋势分析报告[/]")
    console.print(f"  分析时间: {report.analyzed_at}")
    console.print(f"  热门话题: {len(report.hot_topics)} 个")
    for topic in report.hot_topics[:5]:
        console.print(f"    - {topic.keyword} (搜索量: {topic.search_volume}, 增长: {topic.growth_rate:+.1%})")

    console.print(f"\n  爆款规律: {len(report.viral_patterns)} 条")
    for pattern in report.viral_patterns[:5]:
        console.print(f"    - {pattern.pattern_name}: {pattern.description} ({pattern.frequency:.0%})")

    console.print(f"\n  钩子公式: {len(report.hook_formulas)} 个")
    for hook in report.hook_formulas[:5]:
        console.print(f"    - [{hook.hook_type}] {hook.template} (效果: {hook.effectiveness:.0%})")

    # 保存
    if output:
        output_path = Path(output)
    else:
        output_path = OUTPUT_DIR / f"trend_report_{niche}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    )
    console.print(f"\n[green]✓ 报告已保存: {output_path}[/]")

    # 更新钩子库
    hooks_path = Path("skills/hooks-library.md")
    if hooks_path.parent.exists():
        analyzer.update_hooks_library(report, hooks_path)
        console.print(f"[green]✓ 钩子库已更新: {hooks_path}[/]")


@cli.command()
@click.option("--query", "-q", required=True, help="检索查询")
@click.option("--destination", "-d", default=None, help="目的地过滤")
@click.option("--top-k", "-k", default=20, help="返回数量")
@click.option("--output", "-o", default=None, help="输出文件路径")
def retrieve(query: str, destination: str | None, top_k: int, output: str | None):
    """素材语义检索"""
    from tools.retrieve.semantic_asset_retriever import SemanticAssetRetriever

    console.print(f"[cyan]语义检索: '{query}'[/]")
    retriever = SemanticAssetRetriever()
    results = retriever.retrieve(query=query, destination=destination, top_k=top_k)

    table = Table(title="检索结果")
    table.add_column("#", style="dim", width=3)
    table.add_column("素材", style="cyan")
    table.add_column("时间段", style="green")
    table.add_column("相似度", justify="right")
    table.add_column("质量", justify="right")
    table.add_column("摘要", style="dim")

    for i, clip in enumerate(results.clips, 1):
        table.add_row(
            str(i),
            Path(clip.source_path).name,
            f"{clip.start_sec:.1f}-{clip.end_sec:.1f}s",
            f"{clip.score:.3f}",
            f"{clip.quality:.2f}",
            clip.scene_summary[:30],
        )

    console.print(table)

    if output:
        output_path = Path(output)
    else:
        output_path = OUTPUT_DIR / "candidate_clips.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results.model_dump(mode="json"), ensure_ascii=False, indent=2)
    )
    console.print(f"\n[green]✓ 结果已保存: {output_path}[/]")


@cli.command()
@click.argument("pipeline_name")
@click.option("--destination", "-d", default=None, help="目的地")
@click.option("--duration", "-t", type=int, default=None, help="目标时长（秒）")
@click.option("--dry-run", is_flag=True, help="只打印计划不执行")
def run(pipeline_name: str, destination: str | None, duration: int | None, dry_run: bool):
    """运行流水线"""
    pipeline_file = PIPELINE_DEFS_DIR / f"{pipeline_name}.yaml"
    if not pipeline_file.exists():
        console.print(f"[red]✗ 流水线不存在: {pipeline_name}[/]")
        console.print(f"  可用流水线: {', '.join(p.stem for p in PIPELINE_DEFS_DIR.glob('*.yaml'))}")
        sys.exit(1)

    variables = {}
    if destination:
        variables["destination"] = destination
    if duration:
        variables["duration"] = duration

    runner = PipelineRunner()
    runner.run(pipeline_name, variables=variables, dry_run=dry_run)


@cli.command()
@click.option("--video", "-v", required=True, help="视频文件路径")
@click.option("--edit-decision", "-e", default=None, help="剪辑决策 JSON 文件")
@click.option("--gates", "-g", multiple=True, help="指定执行的 gate")
def qa(video: str, edit_decision: str | None, gates: tuple):
    """质量自检"""
    from tools.qa.quality_reviewer import QualityReviewer

    reviewer = QualityReviewer()

    ed = None
    if edit_decision:
        from tools.common.models import EditDecision
        ed_data = json.loads(Path(edit_decision).read_text())
        ed = EditDecision(**ed_data)

    gate_list = list(gates) if gates else None
    review = reviewer.review(video_path=video, edit_decision=ed, gates=gate_list)

    # 打印结果
    status = "[green]✓ 通过[/]" if review.passed else "[red]✗ 不通过[/]"
    console.print(f"\n[bold]质检结果: {status}[/]")
    console.print(f"  总分: {review.overall_score:.2f}")

    table = Table(title="检查项")
    table.add_column("Gate", style="cyan")
    table.add_column("状态", style="green")
    table.add_column("分数", justify="right")
    table.add_column("信息")

    for check in review.checks:
        status = "✓" if check.passed else "✗"
        color = "green" if check.passed else "red"
        table.add_row(
            check.name,
            f"[{color}]{status}[/]",
            f"{check.score:.2f}",
            check.message,
        )

    console.print(table)

    if review.recommendations:
        console.print("\n[yellow]修复建议:[/]")
        for rec in review.recommendations:
            console.print(f"  • {rec}")

    # 保存报告
    report_path = OUTPUT_DIR / "qa_review.json"
    report_path.write_text(
        json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2)
    )
    console.print(f"\n[green]✓ 报告已保存: {report_path}[/]")


@cli.command("list")
@click.option("--destination", "-d", default=None, help="按目的地过滤")
def list_assets(destination: str | None):
    """列出素材库"""
    from tools.common.asset_store import AssetStore

    store = AssetStore()

    if destination:
        assets = store.filter_by_destination(destination)
    else:
        assets = list(store.list_all())

    if not assets:
        console.print("[yellow]素材库为空[/]")
        return

    table = Table(title=f"素材库 ({store.count()} 个)")
    table.add_column("#", style="dim", width=3)
    table.add_column("文件", style="cyan")
    table.add_column("目的地", style="green")
    table.add_column("场景", justify="right")
    table.add_column("时长", justify="right")
    table.add_column("质量", justify="right")
    table.add_column("音频", justify="center")

    for i, asset in enumerate(assets, 1):
        table.add_row(
            str(i),
            Path(asset.source_path).name,
            asset.destination or "-",
            str(len(asset.scenes)),
            f"{asset.metadata.duration:.1f}s",
            f"{asset.metadata.quality_score:.2f}",
            "✓" if asset.metadata.has_audio else "✗",
        )

    console.print(table)


@cli.command("pipelines")
def list_pipelines():
    """列出可用流水线"""
    pipelines = list(PIPELINE_DEFS_DIR.glob("*.yaml"))
    if not pipelines:
        console.print("[yellow]无可用流水线[/]")
        return

    table = Table(title="可用流水线")
    table.add_column("名称", style="cyan")
    table.add_column("描述", style="green")

    for p in pipelines:
        import yaml
        data = yaml.safe_load(p.read_text())
        pipeline = data.get("pipeline", {})
        table.add_row(p.stem, pipeline.get("description", "-"))

    console.print(table)


def main():
    cli()


if __name__ == "__main__":
    main()
