# Travel AutoMontage

> 旅行赛道自动剪辑系统 — Agent 即编排器，YAML 定义流程，Markdown 沉淀领域知识，Python 提供工具层。

---

## 目录

- [核心理念](#核心理念)
- [与 OpenMontage 的差异化](#与-openmontage-的差异化)
- [目录结构](#目录结构)
- [架构总览](#架构总览)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [CLI 命令手册](#cli-命令手册)
- [流水线详解](#流水线详解)
- [模块详解](#模块详解)
- [Skills 知识层](#skills-知识层)
- [Schemas 契约层](#schemas-契约层)
- [数据流](#数据流)
- [开发指南](#开发指南)
- [路线图](#路线图)
- [License](#license)

---

## 核心理念

借鉴 [OpenMontage](https://github.com/) 的三层知识架构，将「执行能力」与「业务知识」彻底解耦：

| 层 | 目录 | 职责 | 格式 |
|---|---|---|---|
| **工具层** | `tools/` | 能做什么 — 可执行的 Python 代码 | `.py` |
| **编排层** | `pipeline_defs/` | 怎么串联 — 流水线定义 | `.yaml` |
| **知识层** | `skills/` | 怎么做好 — 旅行 Vlog 领域规范 | `.md` |
| **技术参考** | `agents_skills/` | 技术细节 — FFmpeg/Remotion/CLIP 用法 | `.md` |
| **契约层** | `schemas/` | 数据校验 — 各阶段输入输出 | `.json` |

Agent（Claude Code / Cursor / WorkBuddy）读 YAML 知道流程，读 skills 知道品质标准，调 tools 执行。

**没有中心化的代码编排器。** Agent 本身就是编排器——Python 只提供工具层和持久化层，所有创意决策、编排逻辑、审查规则都放在 YAML + Markdown 里。

---

## 与 OpenMontage 的差异化

| 维度 | OpenMontage | Travel AutoMontage |
|---|---|---|
| 素材源 | 公开素材库（Pexels / Archive.org） | **自有素材库**（旅行实拍素材语义检索） |
| 趋势感知 | 无 | **抖音热门/主题分析**（巨量算数 + 开放平台 + 人工样本） |
| 领域知识 | 通用视频制作规范 | **旅行赛道专属**：动静结合、景别嵌套、黄金时间轴 |
| 钩子库 | 固定手写 | **数据驱动**：每周自动从爆款提炼新公式，反喂 hooks-library.md |
| 质检门禁 | 通用 | 旅行 Vlog 专项（节奏曲线、目的地一致性、幻灯片风险） |

---

## 目录结构

```
travel-automontage/
├── pipeline_defs/              # YAML: 旅行 Vlog 的 N 条流水线定义
│   ├── travel-vlog.yaml            # 单目的地叙事 Vlog
│   ├── travel-montage.yaml         # 多素材混剪
│   └── travel-shorts.yaml          # 旅行短视频 / Reels
│
├── tools/                      # Python: 可执行工具
│   ├── common/                 # 共享基础设施
│   │   ├── config.py               # 全局配置 & 路径管理
│   │   ├── models.py               # Pydantic 数据模型（Asset, Scene, TrendReport...）
│   │   ├── asset_store.py          # SQLite + 向量持久化层
│   │   └── model_client.py         # 多模态模型客户端（Gemini/Claude）
│   ├── ingest/                 # 素材入库
│   │   └── travel_asset_ingestor.py    # 转码 → AI 分析 → 向量化 → 入库
│   ├── trends/                 # 抖音热门 / 主题分析 ← 核心差异化
│   │   └── douyin_trend_analyzer.py    # 趋势分析 + 爆款拆解 + 钩子库更新
│   ├── retrieve/               # 素材语义检索
│   │   └── semantic_asset_retriever.py # CLIP 向量检索 + 多样性过滤
│   ├── edit/                   # 剪辑决策
│   │   ├── script_generator.py        # 剧本生成（趋势 + 素材 → 钩子/旁白/分镜）
│   │   ├── storyboard_planner.py      # 分镜规划（节拍对齐 / 快剪 / 景别嵌套）
│   │   ├── edit_decision_maker.py     # 剪辑决策（转场/时长/顺序）
│   │   └── bgm_selector.py            # BGM 选择（情绪匹配 + 节奏）
│   ├── compose/                # 渲染合成
│   │   └── ffmpeg_composer.py         # FFmpeg 渲染（截取/拼接/字幕/BGM/调色）
│   ├── qa/                     # 质量自检
│   │   └── quality_reviewer.py        # 8 项 Gate 检查 + 自动回退
│   ├── pipeline_runner.py      # YAML 流水线执行器
│   └── cli.py                  # CLI 入口（tam 命令）
│
├── skills/                     # Markdown: 旅行赛道剪辑规范与品质基准
│   ├── travel-storytelling.md      # 旅行叙事的最佳实践
│   ├── travel-pacing.md            # 旅行 Vlog 节奏规范（动静结合 / 黄金时间轴）
│   ├── douyin-travel-format.md     # 抖音旅行赛道的格式要求
│   └── hooks-library.md            # 开头钩子公式库（数据驱动，自动更新）
│
├── agents_skills/              # Markdown: 外部技术深度知识
│   ├── ffmpeg.md                   # FFmpeg 命令模板速查
│   ├── remotion.md                 # Remotion React 组件用法
│   └── clap-model.md               # CLIP 视频语义理解模型用法
│
├── schemas/                    # JSON: 各阶段输入输出校验
│   ├── asset.schema.json           # 素材资产结构
│   ├── trend_report.schema.json    # 趋势报告结构
│   ├── candidate_clips.schema.json # 候选素材列表结构
│   ├── script.schema.json          # 剧本结构
│   ├── edit_decision.schema.json   # 剪辑决策结构
│   └── qa_review.schema.json       # 质检报告结构
│
├── data/                       # 运行时数据（gitignore）
│   ├── assets_db/                  # SQLite 素材库
│   ├── cache/                      # 模型上传缓存 / 转码缓存
│   ├── output/                     # 最终视频输出
│   └── samples/                    # 样本素材
│
├── tests/                      # 测试
├── pyproject.toml              # 项目依赖配置
├── .gitignore
└── README.md                   # 本文件
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    Agent (编排器)                         │
│         读取 YAML 流程 + Skills 规范 → 调用 Tools         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ YAML 编排层 │  │ MD 知识层  │  │ JSON 契约层│
  │ pipeline_  │  │ skills/   │  │ schemas/  │
  │ defs/      │  │ agents_   │  │           │
  │            │  │ skills/   │  │           │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │                  Python 工具层                        │
  │                                                     │
  │  ingest → trends → retrieve → edit → compose → qa   │
  │                                                     │
  │  common/ (AssetStore, ModelClient, Config, Models)  │
  └─────────────────────────────────────────────────────┘
```

**数据流方向**：素材入库 → 趋势分析 → 语义检索 → 剧本生成 → 分镜规划 → 剪辑决策 → 渲染合成 → 质量自检

---

## 快速开始

### 1. 安装

```bash
cd travel-automontage
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
export GEMINI_API_KEY="your-gemini-api-key"    # 视频AI分析（可选，缺失时回退到ffprobe基础分析）
export DOUYIN_OPEN_TOKEN="your-open-api-token"  # 抖音开放平台（可选，缺失时使用内置默认趋势数据）
```

> 不配置也能跑——所有模块都有降级方案，核心流程不会中断。

### 3. 入库素材

```bash
# 扫描素材目录 → 转码 → AI分析 → 向量化 → 写入SQLite
tam ingest --footage-dir ~/Travel/Fuji2024
```

入库后会生成结构化的素材资产：

```json
{
  "asset_id": "fuji_sunrise_001",
  "destination": "日本-富士山",
  "scenes": [
    {
      "start": "00:00", "end": "00:30",
      "summary": "富士山日出全景，金色时刻光线",
      "visual_tags": ["自然", "山脉", "金色时刻", "全景"],
      "motion_tags": ["静态", "三脚架固定"],
      "audio_tags": ["风噪", "无人声"],
      "quality": 0.92
    }
  ]
}
```

### 4. 分析趋势

```bash
# 分析抖音旅行赛道趋势，提取爆款规律
tam trends --niche 旅行 --destination 富士山
```

输出 `trend_report.json`，包含热门话题、爆款规律、钩子公式、节奏规则、BGM 趋势。

### 5. 运行完整流水线

```bash
# 一键运行：趋势分析 → 素材检索 → 剧本生成 → 剪辑 → 渲染 → 质检
tam run travel-vlog --destination 富士山 --duration 30
```

### 6. 其他常用命令

```bash
# 语义检索素材
tam retrieve --query "富士山日出 金色时刻" --destination 富士山

# 质检已有视频
tam qa --video output/final.mp4

# 查看素材库
tam list

# 查看流水线执行计划（不实际执行）
tam run travel-vlog --destination 富士山 --dry-run
```

---

## 环境配置

### 必需依赖

| 依赖 | 用途 | 必须性 |
|---|---|---|
| Python ≥ 3.11 | 运行时 | 必须 |
| FFmpeg | 视频转码/渲染/分析 | 必须（系统安装） |
| pydantic | 数据模型校验 | 必须（pip安装） |
| PyYAML | YAML 流水线解析 | 必须（pip安装） |
| click | CLI 框架 | 必须（pip安装） |
| rich | 终端美化输出 | 必须（pip安装） |

### 可选依赖

| 依赖 | 用途 | 缺失时降级方案 |
|---|---|---|
| google-generativeai | Gemini 视频AI分析 | 回退到 ffprobe 基础分析 |
| torch + clip | CLIP 语义向量化 | 回退到关键词匹配检索 |
| numpy | 向量运算 | 需要安装才能使用语义检索 |

### 环境变量

| 变量名 | 用途 | 默认值 |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API 密钥 | 无（回退到基础分析） |
| `DOUYIN_OPEN_TOKEN` | 抖音开放平台 OAuth Token | 无（使用内置趋势数据） |
| `TAM_DATA_DIR` | 数据目录路径 | `./data` |
| `TAM_OUTPUT_DIR` | 输出目录路径 | `./data/output` |

### 安装 FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# 验证
ffmpeg -version
```

---

## CLI 命令手册

### `tam ingest` — 素材入库

```bash
tam ingest --footage-dir <DIR> [--destination <NAME>] [--skip-transcode]
```

扫描素材目录，对每个视频文件执行：转码 → 上传模型 API（路径键缓存避免重复上传）→ 逐段分析生成场景摘要 → CLIP 向量化 → 写入 SQLite 素材库。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--footage-dir` | string | 素材目录路径（必填） |
| `--destination` | string | 手动指定目的地（默认自动识别） |
| `--skip-transcode` | flag | 跳过转码步骤（素材已是标准格式时使用） |

### `tam trends` — 趋势分析

```bash
tam trends --niche <NAME> [--destination <NAME>] [--refresh-hooks]
```

分析抖音旅行赛道趋势，输出 `trend_report.json`。如果指定 `--refresh-hooks`，会将提取的钩子公式自动写入 `skills/hooks-library.md`。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--niche` | string | 赛道名称，如"旅行"、"美食旅行"（必填） |
| `--destination` | string | 目的地筛选 |
| `--refresh-hooks` | flag | 自动更新 hooks-library.md |

### `tam retrieve` — 语义检索

```bash
tam retrieve --query <TEXT> [--destination <NAME>] [--top-k <N>]
```

基于 CLIP 向量的语义检索。从素材库中按目的地过滤 → 按余弦相似度排序 → 多样性 Top-K 返回。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--query` | string | 检索文本，如"富士山日出 金色时刻"（必填） |
| `--destination` | string | 目的地过滤 |
| `--top-k` | int | 返回数量（默认 20） |

### `tam run` — 运行流水线

```bash
tam run <PIPELINE> [--destination <NAME>] [--duration <SEC>] [--dry-run]
```

按 YAML 流水线定义依次执行各 stage。每个 stage 的输出作为下一个 stage 的输入。

| 参数 | 类型 | 说明 |
|---|---|---|
| `<PIPELINE>` | string | 流水线名称（travel-vlog / travel-montage / travel-shorts） |
| `--destination` | string | 目的地 |
| `--duration` | int | 目标视频时长（秒） |
| `--dry-run` | flag | 只打印执行计划，不实际运行 |

### `tam qa` — 质量自检

```bash
tam qa --video <PATH> [--strict]
```

对最终视频执行 8 项 Gate 检查，任何一项不通过 → 回到 edit 阶段重做。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--video` | string | 视频文件路径（必填） |
| `--strict` | flag | 严格模式（所有 Gate 必须通过） |

### `tam list` — 查看素材库

```bash
tam list [--destination <NAME>] [--tag <TAG>]
```

列出素材库中的所有素材资产。

---

## 流水线详解

### travel-vlog.yaml — 单目的地叙事 Vlog

```
trend_research → asset_retrieval → script_generation → storyboard → edit_decision → compose → qa
```

7 个 stage，适合制作 30-60 秒的单目的地旅行 Vlog。核心是"趋势驱动 + 素材驱动"双输入的剧本生成。

### travel-montage.yaml — 多素材混剪

```
asset_retrieval → music_selection → montage_planning → compose → qa
```

5 个 stage，适合将多段素材混剪为节奏感强的蒙太奇短片。跳过趋势分析和剧本生成，侧重素材间的视觉节奏。

### travel-shorts.yaml — 旅行短视频 / Reels

```
trend_research → asset_retrieval → script_generation → compose → qa
```

5 个 stage，精简版流水线，适合快速产出 15-30 秒的短视频。合并了 storyboard 和 edit_decision 为一步。

### 流水线 YAML 结构

```yaml
pipeline:
  name: travel-vlog
  description: "单目的地叙事 Vlog"
  variables:
    destination: { required: true, type: string }
    duration: { default: 30, type: int }

  stages:
    - name: trend_research
      tool: trends.douyin_trend_analyzer
      input:
        niche: "旅行"
        destination: "{{destination}}"
      output: trend_report.json
      on_failure: warn          # fatal | warn | retry

    - name: asset_retrieval
      tool: retrieve.semantic_asset_retriever
      input:
        destination: "{{destination}}"
        style: "{{trend_report.hook_type}}"
        count: 20
      output: candidate_clips.json
      depends_on: [trend_research]

    - name: script_generation
      tool: edit.script_generator
      skill: skills/travel-storytelling.md
      input:
        trend: "{{trend_report}}"
        clips: "{{candidate_clips}}"
        duration: "{{duration}}"
      output: script.json
      depends_on: [trend_research, asset_retrieval]

    - name: compose
      tool: compose.ffmpeg_composer
      skill: skills/travel-pacing.md
      output: final_video.mp4
      depends_on: [edit_decision]

    - name: qa
      tool: qa.quality_reviewer
      skill: skills/douyin-travel-format.md
      gates:
        - slideshow_risk_check
        - audio_level_check
        - subtitle_presence_check
        - douyin_format_check
        - pacing_curve_check
        - hook_strength_check
      on_failure: retry          # 质检不过 → 回到 edit 阶段重做
```

---

## 模块详解

### 1. ingest — 素材资产化

**解决什么问题**：把"杂乱的旅行素材"变成"可被语义检索的资产库"。

**流程**：扫描素材目录 → 转码为统一格式（H.264, 1080p） → 上传多模态模型（路径键缓存避免重复上传） → 逐段分析生成结构化场景摘要 → CLIP 向量化 → 写入 SQLite 素材库。

**关键设计**：
- 路径键缓存：已上传过的文件不重复上传，节省 API 调用
- 逐段分析：不是整段丢给模型，而是按场景切分后逐段分析，精度更高
- 自动识别目的地：从场景摘要中提取地名，无需手动标注

### 2. trends — 抖音趋势分析（核心差异化）

**解决什么问题**：OpenMontage 完全没有的模块——让系统从"闭门造车"变成"数据驱动"。

**合规路径**：以巨量算数 + 抖音开放平台为主，第三方数据（蝉妈妈/飞瓜）作补充。不使用爬虫方案。

**三步分析**：
1. 从巨量算数获取关键词热度（搜索量、内容量、增长率）
2. 从热门视频榜获取 Top 视频（需开放平台权限）
3. 对采样爆款视频做 AI 深度分析（钩子类型、结构、情绪曲线、节奏、BGM）

**输出物**：`trend_report.json` — 包含热门话题、爆款规律、钩子公式、节奏规则、BGM 趋势。

**关键设计**：输出会反向喂给 `skills/hooks-library.md`，让钩子库从"固定手写"变成"每周自动从真实爆款提炼"。

### 3. retrieve — 素材语义检索

**解决什么问题**：把"趋势分析结果"和"自有素材库"桥接起来。

**流程**：向量化查询文本 → 按目的地过滤候选素材 → 按余弦相似度排序 → 多样性 Top-K 返回（避免全是同场景）。

**降级方案**：未安装 CLIP/torch 时，回退到关键词匹配检索。

### 4. edit — 剧本生成 + 剪辑决策

四个子模块协作：

| 子模块 | 职责 |
|---|---|
| `script_generator` | 结合趋势报告 + 候选素材，生成完整剧本（标题、钩子、旁白、分镜、BGM建议） |
| `storyboard_planner` | 将剧本转为分镜表（节拍对齐、快剪标记、景别嵌套） |
| `edit_decision_maker` | 将分镜转为剪辑决策（每个片段的源素材、入点出点、转场类型、时长） |
| `bgm_selector` | 根据情绪和节奏选择 BGM（epic/calm/energetic/melancholic/cheerful） |

### 5. compose — 渲染合成

FFmpeg 渲染引擎，读取 `edit_decision.json`：

1. 逐段截取源素材的指定片段
2. 拼接所有片段（应用转场：cut / fade / dissolve）
3. 烧录字幕（指定字体、位置、样式）
4. 混入 BGM（自动对齐时长、调整音量）
5. 输出抖音格式（9:16, 1080x1920, H.264）

### 6. qa — 质量自检

8 项 Gate 检查：

| Gate | 检查内容 | 不通过后果 |
|---|---|---|
| `douyin_format_check` | 9:16 竖屏、≤60s、1080p | 回退重做 |
| `audio_level_check` | 音频电平在 -23 ~ -16 LUFS | 回退重做 |
| `subtitle_presence_check` | 字幕存在且可读 | 回退重做 |
| `slideshow_risk_check` | 不是"图片轮播假视频" | 回退重做 |
| `black_frame_check` | 无黑帧 | 回退重做 |
| `pacing_curve_check` | 动静比合理（有呼吸感） | 警告 |
| `hook_strength_check` | 开头 3 秒有冲击力 | 警告 |
| `info_density_check` | 信息密度达标 | 警告 |

---

## Skills 知识层

Agent 在执行流水线时，会读取对应的 skill 文件来获取"怎样才算做好"的领域知识。

| 文件 | 用途 | 被哪个 stage 引用 |
|---|---|---|
| `travel-storytelling.md` | 旅行叙事结构、情绪曲线设计、目的地呈现节奏 | script_generation |
| `travel-pacing.md` | 动静结合原则、黄金时间轴、景别嵌套、剪辑密度曲线 | edit_decision, compose |
| `douyin-travel-format.md` | 抖音格式硬性要求（9:16, 1080p, ≤60s, 字幕规范） | qa |
| `hooks-library.md` | 开头钩子公式库（数据驱动，自动更新） | script_generation |

### hooks-library.md 的特殊机制

这个文件不是手写的——它是 `douyin_trend_analyzer` 的输出。每次运行 `tam trends --refresh-hooks` 时，分析器会从最新爆款视频中提炼钩子公式，自动更新这个文件。这让钩子库始终保持新鲜，不会被固定公式局限。

---

## Schemas 契约层

每个 stage 的输入输出都有 JSON Schema 校验，确保数据流转有"合同"。

| Schema | 校验对象 |
|---|---|
| `asset.schema.json` | 素材资产（asset_id, destination, scenes[], embeddings, metadata） |
| `trend_report.schema.json` | 趋势报告（hot_topics, viral_patterns, hook_formulas, pacing_rules, music_trends） |
| `candidate_clips.schema.json` | 候选素材列表（clips[], total_count, query, destination） |
| `script.schema.json` | 剧本（title, hook, scenes[], bgm_suggestion, total_duration） |
| `edit_decision.schema.json` | 剪辑决策（timeline[], total_duration, transitions, bgm） |
| `qa_review.schema.json` | 质检报告（passed, gate_results[], overall_score, suggestions[]） |

---

## 数据流

```
素材目录                    抖音趋势
    │                          │
    ▼                          ▼
┌─────────┐            ┌─────────────┐
│ ingest  │            │   trends    │
│ 转码+分析 │            │  爆款拆解    │
│ +向量化  │            │  +钩子提炼   │
└────┬────┘            └──────┬──────┘
     │                        │
     ▼                        │
┌──────────┐                  │
│AssetStore│◄─────────────────┤
│ SQLite   │                  │
└────┬─────┘                  │
     │                        │
     ▼                        ▼
┌─────────────────────────────────┐
│         retrieve (语义检索)       │
│   query + destination → Top-K    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│       edit (剧本+分镜+剪辑决策)    │
│  trend + clips → script.json    │
│  script → storyboard → edit_dec │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         compose (FFmpeg)         │
│  edit_decision → final.mp4      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│            qa (8 Gates)          │
│  不通过 → 回到 edit 重做          │
└─────────────────────────────────┘
```

---

## 开发指南

### 项目结构约定

- **新增工具**：在 `tools/<module>/` 下新建 `.py` 文件，实现核心类
- **新增流水线**：在 `pipeline_defs/` 下新建 `.yaml`，定义 stages
- **新增技能**：在 `skills/` 下新建 `.md`，编写领域规范
- **新增契约**：在 `schemas/` 下新建 `.json`，定义输入输出结构

### 运行测试

```bash
# 语法检查
python -m py_compile tools/**/*.py

# 格式校验
python -c "import json, yaml; [json.load(open(f)) for f in glob.glob('schemas/*.json')]"
python -c "import yaml; [yaml.safe_load(open(f)) for f in glob.glob('pipeline_defs/*.yaml')]"

# 集成测试（无网络，使用默认数据）
python -c "
import sys; sys.path.insert(0, '.')
from tools.trends.douyin_trend_analyzer import DouyinTrendAnalyzer
from tools.edit.script_generator import ScriptGenerator
from tools.edit.bgm_selector import BGMSelector
from tools.common.asset_store import AssetStore

analyzer = DouyinTrendAnalyzer()
report = analyzer.analyze_travel_trends(niche='旅行', destination='日本-富士山')
print(f'TrendReport: {len(report.hot_topics)} topics')

gen = ScriptGenerator()
script = gen.generate(trend=report, clips={'clips': []}, destination='日本-富士山', duration=30)
print(f'Script: \"{script.title}\", {len(script.scenes)} scenes')
"
```

### 代码风格

- 数据模型统一用 Pydantic v2 (`tools/common/models.py`)
- 配置统一走 `tools/common/config.py`，支持环境变量覆盖
- 每个工具模块对外暴露一个核心类，方法签名清晰
- 所有 IO 操作走 `data/` 目录，不污染项目根目录

### 添加自定义流水线

```yaml
# pipeline_defs/my-pipeline.yaml
pipeline:
  name: my-pipeline
  description: "自定义流水线"
  variables:
    destination: { required: true, type: string }

  stages:
    - name: my_stage
      tool: mymodule.my_tool        # 对应 tools/mymodule/my_tool.py
      skill: skills/my-skill.md     # 可选：引用技能文件
      input:
        key: "{{destination}}"
      output: my_output.json
      on_failure: warn              # fatal | warn | retry
```

---

## 路线图

- [x] 项目骨架 & 三层知识架构
- [x] 素材入库（转码 + AI 分析 + 向量化）
- [x] 抖音趋势分析（巨量算数 + 开放平台 + 人工样本）
- [x] 语义检索（CLIP 向量 + 多样性过滤）
- [x] 剧本生成 + 分镜规划 + 剪辑决策
- [x] FFmpeg 渲染合成
- [x] 8 项 Gate 质量自检
- [x] YAML 流水线执行器 + CLI
- [ ] Remotion 渲染引擎（Kinetic Typography 动画）
- [ ] 抖音开放平台 OAuth 实时趋势获取
- [ ] 自动发布到抖音/小红书
- [ ] Web UI 可视化编排界面
- [ ] 多账号矩阵管理

---

## License

MIT
