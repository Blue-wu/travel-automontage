# 素材检索解析层改造方案 v1

> 目标:把「素材入库 → 解析 → 检索」这一段重做,让检索出来的片段真正"贴切"。
> 范围:仅 `ingest` / `retrieve` / `common` 三块 + 新增 `eval`。不动 `edit` / `compose` / `qa`。
> 目标硬件:Windows + i7-8700 + GTX1660 6G(批处理机);MacBook Air M3 16G(检索/创作机)。
> 目标量级:单次约 100–150 段旅行原片,总时长 1–2 小时。

---

## 0. 现状问题清单

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| P1 | 英文 CLIP(`ViT-L/14`)吃中文 query | `config.py:30` | 检索退化成近似随机排序,**头号问题** |
| P2 | 一个中点帧代表整段 | `travel_asset_ingestor.py:175` | 运动、过程、后半段内容全丢 |
| P3 | 固定 10 秒切块 | `model_client.py:147` | 一块跨多个内容单元,标签和向量都不代表它 |
| P4 | 无 `景别` / `运镜` 结构化字段 | `models.py:11` | `travel-pacing.md` 的约束在数据层无法表达 |
| P5 | 无 `可用区间`(起幅落幅) | — | 成片包含甩镜头、失焦的废帧 |
| P6 | `asset_id` = hash(绝对路径\|size\|mtime) | `travel_asset_ingestor.py:115` | 百度云重下一次 → 全库失效重算 |
| P7 | `source_path` 存绝对路径 | `models.py:37` | 库文件在两台机器间不通用 |
| P8 | 检索全量 load 进内存 + N+1 查询 | `semantic_asset_retriever.py:69,82` | 慢,且无法做结构化预过滤 |
| P9 | 关键词回退用 `split()` 切中文 | `semantic_asset_retriever.py:207` | 中文回退路径完全失效 |
| P10 | `gemini-1.5-pro` 已退役 | `config.py:25` | VLM 路径直接报错,静默降级到 ffprobe |
| P11 | 无评测手段 | — | 改了不知道有没有变好 |

---

## 1. 目标架构

```
原片(百度云下载到本地)
  │
  ├─[1] probe      ffprobe + EXIF/GPS + 内容哈希
  │
  ├─[2] proxy      转 720p/5fps 低码率代理文件(仅供上传)
  │
  ├─[3] vlm_tag    整段代理文件 → Gemini 2.5 Pro(原生视频+音频)
  │                └→ 带时间戳的分段 + 结构化标签  ★ 同时完成分段和打标
  │
  ├─[4] signals    本地 CPU:光流运动量化 / 抖动 / 锐度 / 曝光 / 色彩
  │
  ├─[5] embed      本地 GPU:Chinese-CLIP 多帧视觉向量 + BGE 文本向量
  │
  └─[6] store      SQLite(assets / segments / segment_vectors / segments_fts)
                          │
                          ▼
            retrieve: 结构化预过滤 → 三路召回 → RRF → 约束式选片
```

**分工原则**

| 交给 Gemini | 留在本地 |
|---|---|
| 时序分段、语义描述、景别、运镜方向、时段、音频内容、可用性判断 | 运动强度、抖动、锐度、曝光、色彩(要**连续可比数值**,VLM 给不准) |
| 一次性成本,¥20/全量 | 向量化(要反复重算、反复 A/B,必须免费) |

**明确放弃的方案**(上一轮提过,这里撤回):
- ~~PySceneDetect / TransNetV2~~ — 原片是连续拍摄,没有转场可切,切不出东西
- ~~帧网格拼图压 token~~ — 省几块钱,损失时序和音频,不划算
- ~~本地 Whisper + CLAP~~ — Gemini 原生视频带音频理解,免费附送
- ~~1660 上跑 VLM~~ — 6GB + 无 Tensor Core + 不支持 bf16,必爆且无意义

---

## 2. 受控词表(新增 `tools/common/vocab.py`)

**这是整个方案的地基。** VLM 输出、SQL 过滤、`skills/travel-pacing.md` 的约束,三者必须共用同一套词表,否则打标再详细也无法被消费。

```python
"""受控词表 — VLM 输出 / DB 过滤 / skills 约束三者共用"""

SHOT_SCALE = ["ELS", "LS", "MS", "CU", "ECU"]
# ELS 大远景(航拍/超广) LS 远景 MS 中景 CU 近景 ECU 大特写

CAMERA_MOTION = [
    "static",      # 固定机位
    "pan",         # 水平摇
    "tilt",        # 垂直摇
    "push_in",     # 推
    "pull_out",    # 拉
    "tracking",    # 跟拍/移动
    "handheld",    # 手持晃动为主
    "aerial",      # 航拍飞行
]
# 动静二分:travel-pacing.md 的「动静结合」用这个
STATIC_MOTIONS = {"static", "push_in", "pull_out"}
DYNAMIC_MOTIONS = {"pan", "tilt", "tracking", "handheld", "aerial"}

TIME_OF_DAY = ["sunrise", "golden_hour", "midday", "overcast",
               "blue_hour", "night", "indoor", "unknown"]

SCENE_TYPE = ["自然风光", "城市街景", "人文生活", "美食", "交通在途",
              "住宿", "人物", "室内空间", "特写细节"]

MOOD = ["epic", "calm", "lively", "curious", "melancholic", "cheerful", "neutral"]

DEFECT = ["过曝", "欠曝", "失焦", "严重抖动", "穿帮", "画面遮挡", "画质差"]

WEATHER = ["晴", "多云", "阴", "雨", "雪", "雾", "unknown"]


def motion_class(camera_motion: str) -> str:
    """归一到动/静二分,供 pacing 约束使用"""
    return "static" if camera_motion in STATIC_MOTIONS else "dynamic"
```

> 改 `skills/travel-pacing.md` 时,景别和运镜的写法必须用这里的枚举值,否则 Agent 读了也没法映射到查询条件。

---

## 3. 数据模型改造(重写 `tools/common/models.py`)

`Scene` 改名为 `ShotSegment`,语义从"视频的一个场景"变成"**一个可直接上时间线的素材单元**"。

```python
class ShotSegment(BaseModel):
    """一个可直接上时间线的素材单元"""
    segment_id: str            # f"{asset_id}#{idx:03d}"
    asset_id: str
    idx: int

    # ── 时间 ──
    start_sec: float
    end_sec: float
    usable_start_sec: float    # 去掉起幅后的入点  ★ 直接决定成片是否毛糙
    usable_end_sec: float      # 去掉落幅后的出点

    # ── VLM 语义层 ──
    summary: str = ""                              # 中文一句话,20-40 字
    subjects: list[str] = Field(default_factory=list)
    scene_type: str = ""                           # vocab.SCENE_TYPE
    shot_scale: str = ""                           # vocab.SHOT_SCALE
    camera_motion: str = ""                        # vocab.CAMERA_MOTION
    time_of_day: str = "unknown"                   # vocab.TIME_OF_DAY
    weather: str = "unknown"
    mood: str = "neutral"
    has_person: bool = False
    has_speech: bool = False
    ambient_sound: list[str] = Field(default_factory=list)
    vlm_usable: bool = True                        # VLM 主观:这段能不能用
    vlm_quality: float = 0.0                       # 0-1
    defects: list[str] = Field(default_factory=list)

    # ── 本地确定性层 ──
    motion_intensity: float = 0.0   # 光流均值归一 0-1
    shake_score: float = 0.0        # 抖动 0-1,越大越抖
    sharpness: float = 0.0          # Laplacian 方差归一 0-1
    brightness: float = 0.0         # 0-1
    exposure_ok: bool = True
    colorfulness: float = 0.0

    # ── 综合 ──
    quality: float = 0.0            # 融合分,见 §6.4
    usable: bool = True

    @property
    def usable_duration(self) -> float:
        return self.usable_end_sec - self.usable_start_sec


class Asset(BaseModel):
    asset_id: str              # ★ 内容哈希,不再含路径/mtime
    rel_path: str              # ★ 相对 library_root,不再存绝对路径
    proxy_path: str = ""
    destination: str = ""
    gps: tuple[float, float] | None = None
    shot_at: datetime | None = None
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    segments: list[ShotSegment] = Field(default_factory=list)
    tagger_version: str = ""   # 便于按版本增量重跑
    ingested_at: datetime = Field(default_factory=datetime.now)
```

> 旧的 `Scene` / `CandidateClip` 保留一层兼容适配(`ShotSegment.to_candidate_clip()`),避免下游 `edit/` 一次性大改。

---

## 4. SQLite 改造(重写 `tools/common/asset_store.py`)

### 4.1 新 DDL

```sql
CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,          -- 内容哈希
    rel_path       TEXT NOT NULL UNIQUE,      -- 相对 library_root
    proxy_path     TEXT,
    destination    TEXT,
    gps_lat        REAL,
    gps_lon        REAL,
    shot_at        TIMESTAMP,
    duration       REAL,
    width          INTEGER,
    height         INTEGER,
    fps            REAL,
    has_audio      INTEGER,
    tagger_version TEXT,
    ingested_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_assets_dest ON assets(destination);

CREATE TABLE IF NOT EXISTS segments (
    segment_id       TEXT PRIMARY KEY,
    asset_id         TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    start_sec        REAL, end_sec          REAL,
    usable_start_sec REAL, usable_end_sec   REAL,
    summary          TEXT,
    scene_type       TEXT,
    shot_scale       TEXT,
    camera_motion    TEXT,
    motion_class     TEXT,      -- 冗余存,供 pacing 约束直接过滤
    time_of_day      TEXT,
    weather          TEXT,
    mood             TEXT,
    has_person       INTEGER,
    has_speech       INTEGER,
    motion_intensity REAL, shake_score REAL, sharpness REAL,
    brightness       REAL, colorfulness REAL, exposure_ok INTEGER,
    vlm_quality      REAL, quality REAL, usable INTEGER,
    subjects_json    TEXT, ambient_json TEXT, defects_json TEXT,
    FOREIGN KEY(asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_seg_asset  ON segments(asset_id);
CREATE INDEX IF NOT EXISTS idx_seg_filter ON segments(usable, shot_scale, motion_class, time_of_day);
CREATE INDEX IF NOT EXISTS idx_seg_qual   ON segments(quality DESC);

CREATE TABLE IF NOT EXISTS segment_vectors (
    segment_id TEXT NOT NULL,
    kind       TEXT NOT NULL,        -- 'frame' | 'text'
    frame_idx  INTEGER NOT NULL,     -- text 固定 0
    frame_sec  REAL,
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL,        -- float32 raw bytes
    PRIMARY KEY (segment_id, kind, frame_idx)
);
CREATE INDEX IF NOT EXISTS idx_vec_kind ON segment_vectors(kind);

-- 中文全文检索。trigram 分词器对 CJK 有效(SQLite ≥ 3.34)
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    segment_id UNINDEXED,
    summary,
    subjects,
    tokenize = 'trigram'
);
```

> 若 Windows 上 Python 自带的 SQLite 版本不支持 trigram,退路是用 `jieba` 预分词后写入一个 `tokens` 列,tokenizer 用 `unicode61`。入库时判一次 `sqlite3.sqlite_version` 决定走哪条。

### 4.2 关键新查询

```python
def filter_segments(
    self,
    destination: str | None = None,
    shot_scales: list[str] | None = None,
    motion_class: str | None = None,
    time_of_day: list[str] | None = None,
    scene_types: list[str] | None = None,
    min_usable_duration: float = 0.0,
    min_quality: float = 0.0,
    exclude_ids: set[str] | None = None,
    usable_only: bool = True,
) -> list[str]:
    """结构化预过滤,只返回 segment_id 列表。
    这是检索的第一步 —— 把几百条候选先砍到几十条,再做向量计算。"""
```

```python
def load_vectors(self, segment_ids: list[str], kind: str) -> dict[str, np.ndarray]:
    """一次性批量取向量,shape (n_frames, dim)。
    替代原来的逐条 asset_store.get() —— 消除 N+1。"""
```

### 4.3 迁移

旧库缺少全部新增字段,且 `asset_id` 算法变了(路径哈希 → 内容哈希),**无法增量迁移**。

```
1. 备份 data/assets_db/assets.sqlite3 → assets.sqlite3.bak
2. 删除原库,用新 DDL 重建
3. 全量重新入库(100 段素材约 1 小时,见 §8)
```

---

## 5. 入库流水线(新增 `tools/ingest/`)

### 5.1 `hashing.py` — 内容哈希(解决 P6)

```python
import blake3   # pip install blake3;或用 hashlib.blake2b 替代

HEAD_TAIL_BYTES = 8 * 1024 * 1024

def content_hash(path: Path) -> str:
    """读文件头尾各 8MB + 文件大小。
    对重下载、改名、换机器完全免疫,且不需要读完整个大文件。"""
    size = path.stat().st_size
    h = blake3.blake3()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(HEAD_TAIL_BYTES))
        if size > HEAD_TAIL_BYTES * 2:
            f.seek(-HEAD_TAIL_BYTES, 2)
            h.update(f.read(HEAD_TAIL_BYTES))
    return h.hexdigest()[:16]
```

### 5.2 `probe.py` — 元信息 + GPS

```python
def probe(path: Path) -> dict:
    """ffprobe 取 duration/分辨率/fps/音轨。
    注意:原代码用 eval() 解析 r_frame_rate,改成 Fraction 安全解析。"""

def read_gps(path: Path) -> tuple[float, float] | None:
    """从 EXIF/QuickTime metadata 读 GPS。
    手机拍的 mov/mp4 通常在 com.apple.quicktime.location.ISO6709。
    有 GPS 就直接反查地名,比任何模型猜都准,而且免费。"""
```

**先验证一下再实现**:随便挑几个文件跑 `exiftool -G -a -s xxx.mp4 | findstr /i "gps location"`,有输出才值得做这一条。

### 5.3 `proxy.py` — 代理转码(解决上传带宽)

```python
PROXY_SPEC = dict(height=720, fps=5, crf=32, preset="veryfast", abitrate="64k")

def make_proxy(src: Path, dst: Path) -> Path:
    """转 720p/5fps 低码率,仅供上传给 VLM。
    Gemini 默认 1fps 采样,4K 原片纯属浪费带宽。
    45 秒 → 约 5-10MB,100 段总计 <1GB。"""
```

```bash
ffmpeg -i in.mp4 -vf scale=-2:720 -r 5 -c:v libx264 -crf 32 -preset veryfast \
       -c:a aac -b:a 64k -movflags +faststart -y proxy.mp4
```

> ⚠️ 与现有 `_transcode()` 区分开。那个转的是 1080p CRF18(比送 API 需要的质量高得多,文件反而更大),用途是本地抽帧,两条路径各留各的。

### 5.4 `vlm_tagger.py` — 核心模块

**SDK 换代**:`google.generativeai` 已废弃,改用 `google-genai`。

```python
from google import genai
from google.genai import types

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec":        {"type": "number"},
                    "end_sec":          {"type": "number"},
                    "usable_start_sec": {"type": "number"},
                    "usable_end_sec":   {"type": "number"},
                    "summary":          {"type": "string"},
                    "subjects":         {"type": "array", "items": {"type": "string"}},
                    "scene_type":    {"type": "string", "enum": SCENE_TYPE},
                    "shot_scale":    {"type": "string", "enum": SHOT_SCALE},
                    "camera_motion": {"type": "string", "enum": CAMERA_MOTION},
                    "time_of_day":   {"type": "string", "enum": TIME_OF_DAY},
                    "weather":       {"type": "string", "enum": WEATHER},
                    "mood":          {"type": "string", "enum": MOOD},
                    "has_person":    {"type": "boolean"},
                    "has_speech":    {"type": "boolean"},
                    "ambient_sound": {"type": "array", "items": {"type": "string"}},
                    "usable":        {"type": "boolean"},
                    "quality":       {"type": "number"},
                    "defects":       {"type": "array", "items": {"type": "string", "enum": DEFECT}},
                },
                "required": ["start_sec", "end_sec", "usable_start_sec", "usable_end_sec",
                             "summary", "shot_scale", "camera_motion", "usable", "quality"],
            },
        },
    },
    "required": ["segments"],
}
```

用 `response_schema` 强约束输出,**不要再手写 markdown 代码块清洗**(原 `model_client.py:126` 那段 `text.split("\n",1)[1].rsplit("```",1)[0]` 很脆)。

**Prompt**:

```
你在为一个旅行 Vlog 素材库做镜头级标注。这是一段未经剪辑的旅行原片。

任务:把它切成若干"可直接上剪辑时间线的素材单元",每个单元输出结构化标注。

切分规则:
1. 按【内容】和【运镜】的变化切,不是按固定时长切。
   画面主体换了、运镜方式换了、光线环境换了 → 切一刀。
2. 每个单元时长 2-8 秒。超过 8 秒的稳定画面,按 4-6 秒切成多个。
3. usable_start_sec / usable_end_sec 必须**去掉起幅和落幅**:
   开头镜头还没稳、结尾开始甩向别处的部分要排除在外。
   这两个值是剪辑真正会用的入点出点,请严格判断。
4. 完全不可用的段落(严重糊、大幅甩镜、误拍地面、黑屏)
   也要输出,但 usable=false 并在 defects 说明。

标注要求:
- summary:20-40 字中文,描述**画面里有什么**,不要写"这是一个镜头"这类废话。
  好例子:"清晨云海中的富士山山顶,前景是红色鸟居的剪影"
  坏例子:"一个山的远景镜头"
- shot_scale:严格按画面占比判断。航拍/超广角整体环境=ELS,
  全身+环境=LS,腰部以上=MS,胸部以上或物体特写=CU,极致局部=ECU。
- camera_motion:只描述【相机】怎么动,不是画面里的东西怎么动。
- time_of_day:根据光线色温和方向判断。
- has_speech:是否有人在说话(不是环境人声)。
- ambient_sound:环境音标签,如 ["风声","海浪","市集人声","车流"]。
- quality:0-1,综合构图、曝光、稳定度、内容价值给一个**能横向比较**的分数。
  别所有段都给 0.8,要拉开差距。

时间戳以视频起点为 0,单位秒,保留一位小数。
```

**调用**:

```python
def tag_video(self, proxy_path: Path, model: str = "gemini-2.5-flash") -> dict:
    f = self.client.files.upload(file=str(proxy_path))
    resp = self.client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(
                file_uri=f.uri, mime_type="video/mp4",
                # ★ 默认 1fps。旅行空镜变化慢,0.5 通常够用,输入 token 直接减半
                video_metadata=types.VideoMetadata(fps=1.0),
            ),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SEGMENT_SCHEMA,
            temperature=0.2,
            # ★★ 必须显式关闭。默认 -1(动态开启),thinking token 按 output 价计费,
            #    Pro 上会让总成本涨 50%+。本任务是结构化提取 + schema 强约束,
            #    深度思考几乎无增益。
            #    注意:Flash 可设 0;Pro 有最低值(设不到 0 就给最小值)。
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return json.loads(resp.text)
```

**必须保留的工程细节**:
- **结果落盘缓存**,键 = `content_hash + tagger_version`。改 prompt 时 bump version,没改就直接命中缓存不重复花钱。原来的路径键缓存换成内容哈希键。
- Files API 上传的文件约 48 小时过期,缓存要存**解析结果**而不是 file_uri。
- 失败重试 3 次;连续失败则标记该 asset 待处理,不要中断整批。

### 5.4b 本地 VLM 替代路线(零 API 成本)★

`vlm_tagger.py` 必须做成**可插拔 backend**:`gemini` | `qwen_local`。
开发期反复调 prompt 时,云端方案是 ¥6 × N 次;本地方案是一晚上电费。

#### 架构调整:把「分段」从 VLM 手里拿走

原设计让 VLM 一次干两件事。这两件事难度差异极大:

| 任务 | 难点 | 小模型表现 |
|---|---|---|
| 时序分段 + `usable_range` | 要精确到 0.1 秒 | **最弱的一环** |
| 逐段语义标注 | 看几张图填表 | 够用 |

**时序分段改用信号曲线,不用 VLM** —— 免费、更准、可复现:

```
每 0.5s 抽帧 → Chinese-CLIP embedding(§5.6 本来就要算,零额外成本)
  ├─ 相邻帧余弦距离 → 内容变化曲线 c(t)
  └─ Farneback 光流 → 运动强度曲线 m(t)

分段规则:
  m(t) 低 且 c(t) 低         → 稳定可用段,切 3-5 秒素材单元
  m(t) 尖峰                  → 起幅/落幅/甩镜,排除出 usable_range
  c(t) 跃变                  → 内容切换,切一刀
```

拆开后本地 VLM 只需"看 4-6 帧填个表",6GB 显存足够。

#### 模型选型

| 机器 | 模型 | 占用 | 备注 |
|---|---|---|---|
| **台式机 1660 6G** | Qwen3-VL-4B-Instruct **Q4_K_M** (GGUF) | 权重 3–4GB | **推荐**,留 2GB 给 KV + vision token |
| M3 Air 16G | Qwen3-VL-8B **4bit / MLX** | 5–6GB | 质量更好,但 Air 无风扇会降频,适合夜间批处理 |
| ~~1660 + 8B-Q4~~ | ≈6GB | ❌ | 权重占满卡,放不下 vision token |

Qwen3-VL 有 Interleaved-MRoPE 与 Text-Timestamp Alignment,原生面向视频时序,
同尺寸里对本任务最合适。

**耗时**(500 个素材单元 × 6 帧):1660+4B 约 **1–2 小时**;M3 Air+8B 约 2–4 小时。

#### 效果差距(逐字段)

| 字段 | 本地 4B | Gemini Flash | 差距 | 缓解 |
|---|---|---|---|---|
| `summary` | 好 | 很好 | 小 | — |
| `subjects` / `scene_type` / `mood` | 好 | 好 | 很小 | — |
| `time_of_day` / `weather` | 好 | 好 | 小 | — |
| `shot_scale` | 中 | 好 | 中 | 人体框占比几何校验 |
| `camera_motion` | 中 | 好 | 中 | **以 §5.5 光流为准**,VLM 仅兜底 |
| `usable_range` | 弱 | 中 | 大 | **已改用信号曲线,不再依赖 VLM** |
| `quality` | 弱(分数挤在 0.7-0.8) | 中 | 中 | **用 §6.4 本地画质指标覆盖** |
| 音频 | **无** | 有 | 大 | 见下 |

> 把 VLM 不擅长的都交给确定性算法后,实际剩余差距只有 `summary` 文笔和
> `shot_scale` 判断,这两项差距不大。

#### 音频要补回来

本地纯视觉模型丢失 `has_speech` / `ambient_sound`。走本地路线时**恢复**这两个模块
(云端路线因 Gemini 原生带音频而砍掉,此处加回):

- 人声:`faster-whisper` int8,1660 上 small/medium 流畅,全库几分钟
- 环境音:CLAP 打标签(海浪/风声/市集/车流)——`agents_skills/clap-model.md` 已有文档,一直没接

### 5.5 `signals.py` — 本地确定性信号

对每个 segment 的 `usable_range` 内均匀抽 8–12 帧(降到 480p 再算,快很多):

```python
def compute_signals(video: Path, start: float, end: float) -> dict:
    """返回 motion_intensity / shake_score / sharpness / brightness /
    colorfulness / exposure_ok。纯 CPU,i7-8700 上每段几十毫秒。"""
```

| 指标 | 算法 | 归一方式 |
|---|---|---|
| `motion_intensity` | `cv2.calcOpticalFlowFarneback` 相邻帧光流幅值均值 | 全库分位数归一到 0-1 |
| `shake_score` | 光流全局位移的**高频分量**(减去平滑趋势后的残差) | 同上 |
| `sharpness` | `cv2.Laplacian(gray, CV_64F).var()` | 同上 |
| `brightness` | 灰度均值 / 255 | 直接 |
| `exposure_ok` | 直方图两端 2% 桶占比 < 阈值 | 布尔 |
| `colorfulness` | Hasler-Süsstrunk 色彩度 | 分位数归一 |

> 全库分位数归一很重要:绝对阈值在不同相机/不同天气下完全不可比。入库结束后统一做一遍归一化再写 `quality`。

### 5.6 `embedder.py` — 向量化

```python
# 视觉:Chinese-CLIP(解决 P1)
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
model, preprocess = load_from_name("ViT-L-14", device="cuda", download_root="./models")
# fp16 约 0.9GB,1660 上一万张帧几分钟

# 文本:BGE 中文(用于 summary 的文本-文本检索)
# BAAI/bge-small-zh-v1.5,512 维,CPU 就能跑
```

每个 segment:
- **视觉**:在 `usable_range` 内均匀取 **4 帧**,各自 encode,存 4 条向量(`kind='frame'`)
- **文本**:`summary + " " + " ".join(subjects)` → BGE → 1 条向量(`kind='text'`)

检索时视觉相似度取 **max over frames**(late interaction),**不要平均**——平均会把一个单元里的多个语义抹平。

### 5.7 `pipeline.py` — 编排

```python
def ingest_library(library_root: Path, only_new: bool = True) -> IngestReport:
    """
    for each video:
      1. content_hash  → 命中已入库且 tagger_version 一致则跳过
      2. probe + gps
      3. make_proxy
      4. vlm_tag (带缓存)
      5. 对每个 segment: signals + embed
      6. 写库
    最后统一做一遍全库分位数归一 + quality 融合
    """
```

每步独立 try/except,单个文件失败不影响整批,最后汇总失败清单。

---

## 6. 检索层改造(重写 `tools/retrieve/`)

### 6.1 `query.py` — 结构化查询对象

**接口从"一个字符串"改成"一个分镜位需求"**,这是让检索变贴切的关键。

```python
class ShotQuery(BaseModel):
    text: str                                  # 语义描述
    destination: str | None = None
    shot_scales: list[str] | None = None       # 允许的景别,配合景别嵌套约束
    motion_class: str | None = None            # "static" | "dynamic",配合动静结合
    time_of_day: list[str] | None = None
    scene_types: list[str] | None = None
    min_duration: float = 0.0                  # 这个分镜位需要多长
    min_quality: float = 0.35
    exclude_ids: set[str] = Field(default_factory=set)
    top_k: int = 20
```

### 6.2 `hybrid.py` — 三路召回 + RRF

```python
def search(q: ShotQuery) -> list[ScoredSegment]:
    # 1. SQL 结构化预过滤 → 候选 id 列表(几百 → 几十)
    cand = store.filter_segments(...)
    if not cand: cand = store.filter_segments(...relaxed...)   # 逐级放宽

    # 2. 三路打分
    #    visual:  cn_clip.encode_text(q.text) vs frame 向量,max over frames
    #    textual: bge.encode(q.text)          vs text 向量
    #    lexical: segments_fts MATCH,bm25
    # 3. RRF 融合(k=60)
    #    rrf = Σ 1 / (60 + rank_i)
    # 4. 质量加权
    #    final = rrf * (0.7 + 0.3 * segment.quality)
    # 5. 可选 rerank:候选 <50 时,可上 Qwen3-VL-Reranker-2B
```

RRF 的好处是**不需要调三路之间的权重**,只按各路排名融合,对分数尺度不敏感。几百条数据用 numpy 暴力算即可,无需向量库。

### 6.3 `selector.py` — 约束式选片 ★

**这一步把 `travel-pacing.md` 从文档变成可执行约束。**

现状是每个分镜位独立取 top-1,结果是"每个镜头单看都还行,连起来全是同景别的静止空镜"。

```python
def select_timeline(slots: list[ShotQuery]) -> list[ShotSegment]:
    """对整条时间线做集合选择,而非逐位贪心。

    硬约束:
      - 同一 segment 不重复使用
      - 相邻两镜 shot_scale 不同(快剪段 <1.5s 除外)
      - 任意 10 秒窗口内至少一次 motion_class 切换
      - segment.usable_duration >= slot.min_duration
    软目标(加权求和):
      - Σ 检索得分
      - Σ segment.quality
      - 景别序列接近 ELS→MS→CU→LS 的嵌套模式
      - 素材来源分散(避免连续用同一个源文件)

    实现:beam search,beam width=20。
    10-20 个 slot × 每 slot 20 候选,毫秒级出结果。
    """
```

### 6.4 quality 融合公式

```python
quality = (
    0.40 * vlm_quality        # VLM 的综合判断,权重最高
  + 0.20 * sharpness
  + 0.15 * (1 - shake_score)
  + 0.15 * exposure_score     # exposure_ok ? 1.0 : 0.3
  + 0.10 * colorfulness
)
if not vlm_usable or defects: quality *= 0.3
```

权重先按这个跑,等评测集有数据了再调。

---

## 7. 评测集(新增 `tools/eval/`)—— 必须最先做

**没有这个,后面所有模型选择都是拍脑袋。**

### 7.1 格式 `data/eval/queries.json`

```json
{
  "version": 1,
  "queries": [
    {
      "id": "q001",
      "text": "富士山日出 金色时刻 大远景",
      "destination": "日本-富士山",
      "relevant": ["a3f2c1#002", "a3f2c1#003", "b91c77#000"],
      "partial":  ["c77d02#001"]
    }
  ]
}
```

### 7.2 标注流程

1. 先用**任意**版本的检索器把全库 segment 列出来(带 summary 和缩略图)
2. 挑 **30–50 条你剪片时真会想到的 query**
3. 每条人工标 3–10 个"确实贴切"的 segment_id
4. 半天工作量,一次投入长期复用

### 7.3 指标

```python
# Recall@10, MRR, nDCG@10
# partial 命中算 0.5 分
python -m tools.eval.run_eval --config configs/exp_a.yaml
```

### 7.4 必须跑的对照实验

| 实验 | 变量 | 验证什么 |
|---|---|---|
| A0 | 现状(英文 CLIP + 10s 块 + 单帧) | 基线 |
| A1 | 换 Chinese-CLIP,其余不变 | **P1 单独贡献多少** |
| A2 | A1 + VLM 分段 + 多帧 | **分段和多帧贡献多少** |
| A3 | A2 + 三路混合 + RRF | 检索架构贡献多少 |
| A4 | A3 视觉换 Qwen3-VL-Embedding-2B | 值不值得慢几十倍 |

分开跑才知道钱和时间该花在哪。我的预期是 **A0→A1 的跳变最大**。

---

## 8. 配置改造(`tools/common/config.py`)

```python
@dataclass
class LibraryConfig:
    root: Path = field(default_factory=lambda: Path(os.getenv("TAM_LIBRARY_ROOT", "./footage")))
    # ★ 素材库根目录。DB 里只存相对路径,两台机器各配各的 root(解决 P7)

@dataclass
class VLMConfig:
    provider: str = "gemini"
    model: str = os.getenv("TAM_VLM_MODEL", "gemini-2.5-flash")
    # ★ 不要写死在代码里。原 gemini-1.5-pro 已退役(P10);
    #   gemini-2.5-flash 将于 2026-10-16 下线 —— 到期前换掉
    # ★ 默认走 Flash:全量 ≈¥6,先跑评测集验证质量,不够再切 Pro(≈¥24)
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    thinking_budget: int = 0       # ★ 必须显式 0,默认 -1 会让成本涨 50%+,见 §10.1
    video_fps: float = 1.0         # 降到 0.5 可让输入 token 减半
    tagger_version: str = "v1"     # 改 prompt 就 bump,用于缓存失效(会触发全量重付费)

@dataclass
class EmbedConfig:
    visual_model: str = "ViT-L-14"       # cn_clip
    text_model: str = "BAAI/bge-small-zh-v1.5"
    device: str = os.getenv("TAM_DEVICE", "cuda")
    frames_per_segment: int = 4
```

**两台机器的差异只体现在环境变量里**,`assets.sqlite3` 可以直接拷贝复用:

```
台式机:  TAM_LIBRARY_ROOT=D:\Travel   TAM_DEVICE=cuda
M3 Air:  TAM_LIBRARY_ROOT=/Users/xx/Travel  TAM_DEVICE=cpu
```

---

## 9. 目录结构变化

```
tools/
├── common/
│   ├── config.py          [改] LibraryConfig / VLMConfig / EmbedConfig
│   ├── models.py          [重写] Asset / ShotSegment
│   ├── vocab.py           [新] 受控词表
│   ├── hashing.py         [新] 内容哈希
│   └── asset_store.py     [重写] 新 schema + filter_segments + load_vectors
├── ingest/
│   ├── probe.py           [新]
│   ├── proxy.py           [新]
│   ├── vlm_tagger.py      [新] ★ 核心
│   ├── signals.py         [新]
│   ├── embedder.py        [新]
│   └── pipeline.py        [新] 替代 travel_asset_ingestor.py
├── retrieve/
│   ├── query.py           [新] ShotQuery
│   ├── hybrid.py          [新] 三路召回 + RRF
│   └── selector.py        [新] ★ 约束式选片
└── eval/
    ├── dataset.py         [新]
    └── run_eval.py        [新]

data/eval/queries.json     [新] 评测集
docs/                      [新] 本文档
```

`model_client.py` 拆解:Gemini 部分 → `vlm_tagger.py`,CLIP 部分 → `embedder.py`,ffprobe → `probe.py`,原文件删除。

---

## 10. 实施顺序

| 阶段 | 内容 | 工时 | 花费 | 验收 |
|---|---|---|---|---|
| **S0** | 环境:Windows 装 ffmpeg / PyTorch cu124 / cn_clip / opencv | 0.5d | 0 | `ffmpeg -version` 通 + CUDA 可见 |
| **S1** | `vocab.py` + `models.py` + `hashing.py` + `asset_store.py` 新 schema | 1d | 0 | 空库能建表 |
| **S2** | 评测集标注(30–50 条) | 0.5d | 0 | `queries.json` 就位 |
| **S3** | `embedder.py` 换 Chinese-CLIP,跑实验 **A1** | 1d | 0 | **拿到 A0→A1 提升数字** |
| **S4** | `signals.py` 光流/画质 + **信号曲线分段**(§5.4b),跑实验 **A2** | 1.5d | **0** | 分段带 usable_range,废片被降权 |
| **S5** | `hybrid.py` 三路 + RRF,跑实验 **A3** | 1.5d | 0 | 混合检索优于单路 |
| **S6** | 打标:先本地 Qwen3-VL-4B(档 1),评测不足再切 Gemini(档 2) | 2d | 0 → ¥6 | 结构化字段就位 |
| **S7** | `selector.py` 约束式选片 | 1.5d | 0 | 输出时间线满足景别/动静约束 |

**两个决策点:**
- **S3 结束**:若 A1(仅换 Chinese-CLIP)已把 Recall@10 拉到可接受,后续投入可大幅削减
- **S6 开始前**:先跑本地档 1,只有评测集证明不够才花钱上档 2

> 注意 S4 与 S6 的顺序相比 v1 已调换 —— 分段改由信号曲线完成(§5.4b),
> 不再依赖 VLM,所以 `signals.py` 必须先于打标落地。

单次全量入库耗时估算(100–150 段,1660):
- proxy 转码 ~15 min(CPU 多进程)
- 上传 <1GB ~10 min
- VLM 打标 ~20 min(并发 4)
- 向量化 ~5 min
- signals ~5 min
- **合计约 1 小时**

---

## 10.1 成本明细

按 130 段、总时长 1.5 小时(5400 秒)测算。

**token 构成**

| 项 | 量 |
|---|---|
| 视频 1fps × 258 token/帧 | 1.39M |
| 音频 32 token/秒 | 0.17M |
| prompt × 130 次调用 | 0.08M |
| **输入合计** | **1.64M** |
| JSON 输出(每视频约 6 个 segment) | 0.13M |

**单次全量建库**

| 模型 | thinking=0 | thinking 默认(-1) |
|---|---|---|
| Gemini 2.5 Flash($0.30 / $2.50) | **$0.82 ≈ ¥6** | $1.31 ≈ ¥9 |
| Gemini 2.5 Pro($1.25 / $10) | **$3.35 ≈ ¥24** | $5.30 ≈ ¥38 |

**⚠️ thinking token 按 output 价计费,且 Google 直连 API 默认开启动态思考。**
不显式关掉,Pro 上成本直接涨 58%。见 §5.4 的 `thinking_config`。

**这笔钱什么时候花**

| 场景 | 花费 |
|---|---|
| 首次全量建库 | 见上表 |
| **之后每次检索素材** | **¥0** — 纯本地向量计算,不调 API |
| **之后每剪一条片子** | **¥0** |
| 新增 20 段素材 | 按量比例,Flash ≈ ¥1 / Pro ≈ ¥4 |
| 改了 prompt 想重打全库 | 再付一次全量价(所以 `tagger_version` 要谨慎 bump) |

缓存机制见 §5.4:键 = `content_hash + tagger_version`,只要素材内容和 prompt 版本都没变,永远命中缓存不重复付费。

**降本档位**(按性价比排序)

| 手段 | 效果 | 代价 |
|---|---|---|
| `thinking_budget=0` | 省 30–50% | 无(本任务不需要思考) |
| 用 Flash 替代 Pro | ¥24 → ¥6 | 标注质量下降,需评测集验证 |
| 采样 1fps → 0.5fps | 输入 token 减半 | 快速运镜段的分段精度下降 |
| 关闭音轨 | 省约 11% | 失去 `has_speech` / `ambient_sound` |

### 三档路线(按成本从低到高)

| 档 | 内容 | 花费 | 停在这的条件 |
|---|---|---|---|
| **档 0** | Chinese-CLIP + 信号曲线分段,**完全不用 VLM** | **¥0** | 评测集 Recall@10 已达标 → 收工 |
| **档 1** | 档 0 + Qwen3-VL-4B 本地打标 + Whisper/CLAP(§5.4b) | **¥0**(1–2h/次) | **大概率停在这** |
| **档 2** | 打标换 Gemini Flash + thinking=0 | ¥6/次 | 仅当评测集证明档 1 不足 |
| 档 2+ | 换 Gemini Pro | ¥24/次 | 仅当 Flash 也不足 |

**从档 0 开始,逐档往上,每档跑一次评测集。**

理由:你当前最大的问题是 P1(英文 CLIP 吃中文 query),它在**档 0 就修掉了**,
而且是数量级的变化,和 VLM 打标毫无关系。很可能档 0 之后收益就已经饱和,
连档 1 的一两小时都能省。

**不要靠猜来选模型 —— 这正是 §7 评测集存在的意义。**

### 关于免费额度

Google AI Studio 有免费层(有速率限制),130 段素材不赶时间的话分几天可能免费跑完。

> ⚠️ **隐私**:免费层数据通常会被用于改进产品;付费层不参与训练。
> 旅行素材可能含人脸、住址、行程轨迹,是否上传按个人偏好决定 —— 这不纯粹是成本问题。

---

## 11. 遗留风险

| 风险 | 处置 |
|---|---|
| 原片可能无 EXIF GPS(网盘同步剥离) | S0 阶段先 `exiftool` 抽查,没有就砍掉这条 |
| Windows 自带 SQLite 可能不支持 fts5 trigram | 运行时检测,退路走 jieba 预分词 + unicode61 |
| PyTorch 新版可能已砍 sm_75 | 装之前确认 wheel 支持 Turing;必要时锁 cu124 |
| bf16 checkpoint 在 1660 上 NaN | 加载时显式 `.half()`,不要用 `torch_dtype="auto"` |
| VLM 的 `quality` 分可能全挤在 0.7-0.9 | prompt 里已要求拉开差距;若仍集中,入库后做分位数重标定 |
| VLM 判 `shot_scale` / `camera_motion` 不稳 | 用 signals 的 `motion_intensity` 交叉校验,冲突时以本地数值为准 |

---

## 附:参考

- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing) — 2.5 Pro $1.25/$10 per M,2.5 Flash $0.30/$2.50 per M
- [Vertex AI 视频理解文档](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding) — 原生视频 ~1h(无音频)/~45min(含音频),默认 1fps 采样
- [Chinese-CLIP (OFA-Sys)](https://github.com/OFA-Sys/Chinese-CLIP)
- [QwenLM/Qwen3-VL](https://github.com/qwenlm/qwen3-vl) — 2B/4B/8B/32B dense,Interleaved-MRoPE + Text-Timestamp Alignment
- [Qwen3-VL: How to Run — Unsloth 文档](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) — GGUF/量化部署,4B-Q4 约 3–4GB
- [Best Local Vision Language Models 2026 — TinyWeights](https://tinyweights.dev/posts/best-local-vision-language-models-2026/) — 小 VLM 显存对照
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) / [Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) — A4 实验用
- [BAAI/FlagEmbedding](https://github.com/flagopen/flagembedding) — bge-small-zh-v1.5
