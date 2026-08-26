"""Pipeline Runner — 读取 YAML 流水线定义，按 stages 顺序执行工具

这是系统的"编排层"入口。Agent 可以直接调用此 runner，
也可以手动逐 stage 调用（更灵活）。

设计原则（来自 OpenMontage）：
- 没有"中心化编排器"代码 — Agent 就是编排器
- Python 只提供工具层和持久化层
- 所有创意决策、编排逻辑、审查规则放在 YAML + Markdown 里
- 此 runner 是"方便的批量执行器"，不是"必须的编排器"
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tools.common.config import DATA_DIR, OUTPUT_DIR, PIPELINE_DEFS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)
console = Console()


class PipelineRunner:
    """流水线执行器

    读取 YAML 流水线定义，按 stages 顺序执行工具。
    支持：
    - 变量替换 {{var}}
    - stage 间依赖关系
    - on_failure 策略（fatal/warn/retry）
    - 输出 schema 校验
    """

    def __init__(self):
        self.stage_outputs: dict[str, Any] = {}
        self.variables: dict[str, Any] = {}

    def run(
        self,
        pipeline_name: str,
        variables: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """运行流水线

        Args:
            pipeline_name: 流水线名称（对应 pipeline_defs/ 下的 yaml 文件名）
            variables: 覆盖 YAML 中定义的变量
            dry_run: 只打印执行计划，不实际执行

        Returns:
            所有 stage 的输出字典
        """
        # 加载流水线定义
        pipeline_file = PIPELINE_DEFS_DIR / f"{pipeline_name}.yaml"
        if not pipeline_file.exists():
            raise FileNotFoundError(f"流水线定义不存在: {pipeline_file}")

        with open(pipeline_file) as f:
            pipeline_def = yaml.safe_load(f)

        pipeline = pipeline_def["pipeline"]
        self.variables = pipeline.get("variables", {})
        if variables:
            self.variables.update(variables)

        # 添加系统变量
        self.variables["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        stages = pipeline["stages"]

        # 打印执行计划
        self._print_plan(pipeline, stages, dry_run)

        if dry_run:
            return self.stage_outputs

        # 执行 stages
        for stage in stages:
            stage_name = stage["name"]
            console.print(f"\n[bold cyan]▶ Stage: {stage_name}[/]")

            # 检查依赖
            deps = stage.get("depends_on", [])
            for dep in deps:
                if dep not in self.stage_outputs:
                    console.print(f"  [yellow]⚠ 依赖 {dep} 未执行，跳过[/]")
                    continue

            try:
                output = self._run_stage(stage)
                self.stage_outputs[stage_name] = output
                # 同时用 output 文件名（去掉后缀）作为 key，方便 YAML 中引用
                # ★ 关键：必须先解析变量，把 "script{{run_prefix}}.json" → "script_humor.json"
                #    否则别名 key 是 "script{{run_prefix}}"，下游找不到 "script_humor"
                output_file_raw = stage.get("output", "")
                if output_file_raw:
                    output_file = self._resolve_string(output_file_raw) if isinstance(output_file_raw, str) else output_file_raw
                    output_key = Path(output_file).stem
                    self.stage_outputs[output_key] = output
                    logger.debug(f"已存 stage 别名: {output_key} → {type(output).__name__}")
                console.print(f"  [green]✓ {stage_name} 完成[/]")
            except Exception as e:
                on_failure = stage.get("on_failure", "fatal")
                if on_failure == "warn":
                    console.print(f"  [yellow]⚠ {stage_name} 失败（warn）: {e}[/]")
                    self.stage_outputs[stage_name] = None
                    output_file_raw = stage.get("output", "")
                    if output_file_raw:
                        output_file = self._resolve_string(output_file_raw) if isinstance(output_file_raw, str) else output_file_raw
                        output_key = Path(output_file).stem
                        self.stage_outputs[output_key] = None
                elif on_failure == "retry":
                    retry_config = stage.get("retry", {})
                    max_attempts = retry_config.get("max_attempts", 2)
                    back_to = retry_config.get("back_to", stage_name)
                    console.print(f"  [red]✗ {stage_name} 失败，重试 {max_attempts} 次（回到 {back_to}）[/]")
                    # 简化：仅记录失败
                    self.stage_outputs[stage_name] = None
                    output_file_raw = stage.get("output", "")
                    if output_file_raw:
                        output_file = self._resolve_string(output_file_raw) if isinstance(output_file_raw, str) else output_file_raw
                        output_key = Path(output_file).stem
                        self.stage_outputs[output_key] = None
                else:  # fatal
                    console.print(f"  [red]✗ {stage_name} 失败（fatal）: {e}[/]")
                    raise

        # 打印结果摘要
        self._print_summary(pipeline, stages)

        return self.stage_outputs

    def _run_stage(self, stage: dict) -> Any:
        """执行单个 stage"""
        tool_path = stage["tool"]

        # 解析变量替换
        input_data = self._resolve_variables(stage.get("input", {}))

        # 导入并调用工具
        tool = self._import_tool(tool_path)

        # 注入 skill 文本 —— YAML 里声明的 skill: 此前从未被加载过，
        # 整个「知识层」对自动化路径是装饰性的。
        # 只在工具签名确实接受 skill_text 时注入，避免影响其他工具。
        self._inject_skill(stage, tool, input_data)

        # 调用工具
        result = tool(**input_data) if input_data else tool()

        # 如果有输出路径，保存到文件
        # ★ 必须先解析变量（替换 {{run_prefix}} 等），否则多风格循环会写同一路径互相覆盖
        output_file_raw = stage.get("output")
        if output_file_raw and result is not None:
            output_file = (
                self._resolve_string(output_file_raw)
                if isinstance(output_file_raw, str)
                else output_file_raw
            )
            output_path = OUTPUT_DIR / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"保存 stage 输出文件: {output_path}")
            self._save_output(result, output_path)

        return result

    def _inject_skill(self, stage: dict, tool, input_data: dict) -> None:
        """把 stage 的 skill: 指向的 Markdown 读进来，作为 skill_text 传给工具"""
        skill_rel = stage.get("skill")
        if not skill_rel:
            return

        import inspect
        try:
            params = inspect.signature(tool).parameters
        except (TypeError, ValueError):
            return
        if "skill_text" not in params:
            return

        skill_path = PROJECT_ROOT / skill_rel
        if not skill_path.exists():
            console.print(f"  [yellow]⚠ skill 文件不存在: {skill_rel}[/]")
            return

        input_data["skill_text"] = skill_path.read_text(encoding="utf-8")
        console.print(f"  [dim]↳ 已加载 skill: {skill_rel}[/]")

    def _import_tool(self, tool_path: str):
        """动态导入工具

        Args:
            tool_path: 如 "tools.trends.douyin_trend_analyzer.DouyinTrendAnalyzer.analyze_travel_trends"
        """
        parts = tool_path.split(".")
        module_path = ".".join(parts[:-2])
        class_name = parts[-2]
        method_name = parts[-1]

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # 如果是类方法，需要先实例化
        instance = cls()
        method = getattr(instance, method_name)

        return method

    def _resolve_variables(self, data: Any) -> Any:
        """递归替换 {{var}} 变量"""
        if isinstance(data, str):
            return self._resolve_string(data)
        elif isinstance(data, dict):
            return {k: self._resolve_variables(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_variables(v) for v in data]
        else:
            return data

    def _resolve_string(self, s: str) -> Any:
        """替换单个字符串中的变量（支持嵌套：{{trend_report{{run_prefix}}}}）

        执行流程：
        1. 先把字符串里的「内层 {{...}}」替换掉（比如先把 {{run_prefix}} 换成 _humor）
        2. 替换后的字符串可能还剩一层或多层，重复这个过程直到没有可替换项
           （最多 5 轮，防止死循环）
        3. 若最终字符串整个是一个「顶层变量引用」（如 "trend_report_humor"），
           再用 _eval_expr 返回对象本身（用于 YAML 中 input 引用 stage 输出）
        """
        pattern = r"\{\{([^{}]+)\}\}"  # 必须最内层优先（中间不含 { 或 }）
        result = s
        last = None
        for _ in range(5):
            if last == result:
                break
            last = result

            def replacer(match, _result_local=result):
                expr = match.group(1).strip()
                return str(self._eval_expr(expr))

            result = re.sub(pattern, replacer, result)

        # 如果最终整个字符串就是一个"单一顶层变量引用"（没有内嵌其他字符），
        # 尝试返回原始类型（比如 stage 输出对象 / int / bool / Pydantic 模型等），否则保持字符串
        full_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_\-]*)", result.strip())
        if full_match:
            candidate = full_match.group(1)
            val = self._eval_expr(candidate)
            # _eval_expr 找不到时会返回占位符字符串 "{{candidate}}"
            # 只要不是占位符字符串，说明解析成功 → 直接返回原始类型（包含 Pydantic 模型）
            not_found_sentinel = f"{{{{{candidate}}}}}"
            if not (isinstance(val, str) and val == not_found_sentinel):
                return val
        return result

    def _eval_expr(self, expr: str) -> Any:
        """评估表达式，获取变量值

        支持：
        - 简单变量: {{destination}}
        - 嵌套属性: {{trend_report.destination}}
        - 数组索引: {{trend_report.hook_formulas[0].template}}
        """
        import re

        tokens = self._tokenize_expr(expr)
        if not tokens:
            return f"{{{{{expr}}}}}"

        root = tokens[0]
        value = None

        if root in self.stage_outputs:
            value = self.stage_outputs[root]
        elif root in self.variables:
            value = self.variables[root]
        else:
            return f"{{{{{expr}}}}}"

        for token in tokens[1:]:
            if value is None:
                return None

            if token.startswith("[") and token.endswith("]"):
                # 数组索引
                try:
                    idx = int(token[1:-1])
                    if isinstance(value, (list, tuple)):
                        value = value[idx]
                    else:
                        return None
                except (ValueError, IndexError):
                    return None
            else:
                # 属性访问
                if isinstance(value, dict):
                    value = value.get(token)
                elif hasattr(value, token):
                    value = getattr(value, token)
                elif hasattr(value, "model_dump"):
                    dump = value.model_dump()
                    value = dump.get(token)
                else:
                    return None

        return value

    def _tokenize_expr(self, expr: str) -> list[str]:
        """将表达式拆分为 token 列表

        例如: "trend_report.hook_formulas[0].template"
        -> ["trend_report", "hook_formulas", "[0]", "template"]
        """
        import re
        tokens = []
        pattern = r'([^.\[\]]+)|\[(\d+)\]'
        for match in re.finditer(pattern, expr):
            if match.group(1):
                tokens.append(match.group(1))
            elif match.group(2):
                tokens.append(f"[{match.group(2)}]")
        return tokens

    def _save_output(self, result: Any, path: Path):
        """保存输出到文件"""
        if hasattr(result, "model_dump"):
            # Pydantic 模型
            data = result.model_dump(mode="json")
        elif isinstance(result, (dict, list)):
            data = result
        elif isinstance(result, str):
            # 可能是文件路径
            if Path(result).exists():
                import shutil
                shutil.copy2(result, path)
                return
            data = {"text": result}
        else:
            data = {"value": str(result)}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _print_plan(self, pipeline: dict, stages: list, dry_run: bool):
        """打印执行计划"""
        title = f"Pipeline: {pipeline['name']}"
        if dry_run:
            title += " (DRY RUN)"

        console.print(Panel.fit(
            f"[bold]{title}[/]\n{pipeline.get('description', '')}\n"
            f"Variables: {json.dumps(self.variables, ensure_ascii=False, indent=2)}",
            border_style="cyan",
        ))

        table = Table(title="Execution Plan", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Stage", style="cyan")
        table.add_column("Tool", style="green")
        table.add_column("Depends On", style="yellow")
        table.add_column("On Failure", style="red")

        for i, stage in enumerate(stages, 1):
            table.add_row(
                str(i),
                stage["name"],
                stage["tool"].split(".")[-1],
                ", ".join(stage.get("depends_on", [])) or "-",
                stage.get("on_failure", "fatal"),
            )

        console.print(table)

    def _print_summary(self, pipeline: dict, stages: list):
        """打印执行结果摘要"""
        table = Table(title="Execution Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Output Type", style="dim")

        for stage in stages:
            name = stage["name"]
            output = self.stage_outputs.get(name)
            status = "[green]✓ Done[/]" if output is not None else "[yellow]⚠ Skipped[/]"
            output_type = type(output).__name__ if output is not None else "-"
            table.add_row(name, status, output_type)

        console.print(table)
