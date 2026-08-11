# CLAP / 视频多模态模型参考

> Agent 在 `ingest` 和 `retrieve` 阶段使用多模态模型时的技术参考。
> CLAP (Contrastive Language-Audio Pretraining) 和视频理解模型是素材语义化的核心。

## 1. 模型选型

### 1.1 视频理解模型对比

| 模型 | 能力 | 速度 | 部署方式 | 适用场景 |
|---|---|---|---|---|
| Gemini 1.5 Pro | 视频理解 + 场景描述 + OCR | 快（API） | API | 逐段分析、场景摘要 |
| Claude 3.5 Sonnet | 视频理解 + 创意描述 | 中（API） | API | 高质量场景描述 |
| CLIP (ViT-L/14) | 图像-文本相似度 | 快 | 本地 | 语义检索、向量化 |
| VideoCLIP | 视频-文本相似度 | 中 | 本地 | 视频级语义检索 |
| Whisper | 语音转文字 | 快 | 本地 | 音轨转录 |
| CLAP | 音频-文本相似度 | 快 | 本地 | 音频语义检索 |

### 1.2 推荐组合

```
入库阶段：Gemini 1.5 Pro（场景分析） + CLIP（向量化）
检索阶段：CLIP（语义检索）
趋势分析：Gemini 1.5 Pro（爆款视频拆解）
```

## 2. Gemini File API 用法

### 2.1 上传文件（带路径缓存）

```python
import hashlib
from pathlib import Path
import google.generativeai as genai
import json

class GeminiFileManager:
    """文件上传管理器 — 路径键缓存避免重复上传"""

    CACHE_FILE = "data/cache/gemini_file_cache.json"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        path = Path(self.CACHE_FILE)
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_cache(self):
        Path(self.CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_FILE).write_text(json.dumps(self.cache, indent=2))

    def _file_key(self, video_path: str) -> str:
        """基于文件路径 + 大小 + 修改时间生成缓存键"""
        p = Path(video_path)
        stat = p.stat()
        raw = f"{p.absolute()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def upload(self, video_path: str) -> str:
        """上传视频文件，返回 file URI（带缓存）"""
        key = self._file_key(video_path)
        if key in self.cache:
            return self.cache[key]["uri"]

        # 上传新文件
        file_obj = genai.upload_file(path=video_path)
        self.cache[key] = {
            "uri": file_obj.uri,
            "path": video_path,
        }
        self._save_cache()
        return file_obj.uri

    def delete(self, video_path: str):
        """删除缓存和远端文件"""
        key = self._file_key(video_path)
        if key in self.cache:
            uri = self.cache[key]["uri"]
            try:
                genai.delete_file(uri)
            except Exception:
                pass
            del self.cache[key]
            self._save_cache()
```

### 2.2 逐段分析视频

```python
def analyze_video_scenes(file_uri: str) -> list[dict]:
    """用 Gemini 分析视频，生成结构化场景摘要"""
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = """
    分析这个旅行视频，按场景分段输出 JSON：
    {
      "destination": "识别的目的地",
      "scenes": [
        {
          "start": "HH:MM:SS",
          "end": "HH:MM:SS",
          "summary": "场景描述",
          "visual_tags": ["标签1", "标签2"],
          "motion_tags": ["静态", "手持跟拍", ...],
          "audio_tags": ["人声", "风噪", "BGM", ...],
          "quality": 0.0-1.0
        }
      ]
    }
    只输出 JSON，不要其他文字。
    """

    response = model.generate_content([prompt, {"file_data": {"file_uri": file_uri}}])
    return json.loads(response.text)
```

## 3. CLIP 语义向量化

### 3.1 安装与初始化

```python
# 安装
# pip install torch transformers openai-clip

import torch
import clip
from PIL import Image

class CLIPEncoder:
    """CLIP 编码器 — 将文本和图像转为向量"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)

    def encode_text(self, text: str) -> list[float]:
        """将查询文本转为向量"""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        # L2 归一化
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().tolist()

    def encode_image(self, image_path: str) -> list[float]:
        """将图像转为向量"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().tolist()

    def encode_video_frame(self, video_path: str, timestamp_sec: float) -> list[float]:
        """提取视频指定时间戳的帧并编码"""
        import subprocess
        # 用 ffmpeg 提取帧
        frame_path = f"/tmp/frame_{hash(video_path)}.jpg"
        subprocess.run([
            "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", frame_path, "-y"
        ], check=True, capture_output=True)
        return self.encode_image(frame_path)
```

### 3.2 视频级向量化策略

```python
def embed_video(self, video_path: str, scenes: list[dict]) -> list[list[float]]:
    """对视频的每个场景生成向量"""
    embeddings = []
    for scene in scenes:
        # 取场景中间帧作为代表帧
        mid_sec = (scene["start_sec"] + scene["end_sec"]) / 2
        embedding = self.encode_video_frame(video_path, mid_sec)
        embeddings.append(embedding)
    return embeddings
```

## 4. 语义检索

### 4.1 余弦相似度

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
```

### 4.2 多样性 Top-K

```python
def diverse_top_k(scored_items: list[tuple], k: int, min_diversity: float = 0.3) -> list:
    """多样性 Top-K — 避免返回全部相似场景"""
    result = []
    for item, score in sorted(scored_items, key=lambda x: -x[1]):
        if len(result) >= k:
            break
        # 检查与已选结果的多样性
        is_diverse = True
        for selected, _ in result:
            if hasattr(item, 'embedding') and hasattr(selected, 'embedding'):
                sim = cosine_similarity(item.embedding, selected.embedding)
                if sim > 1 - min_diversity:
                    is_diverse = False
                    break
        if is_diverse:
            result.append((item, score))
    return result
```

## 5. Whisper 音频转录

### 5.1 转录音轨

```python
import whisper

def transcribe_audio(video_path: str) -> dict:
    """提取音轨并用 Whisper 转录"""
    model = whisper.load_model("base")

    # 提取音轨
    import subprocess
    audio_path = "/tmp/audio.wav"
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ], check=True, capture_output=True)

    # 转录
    result = model.transcribe(audio_path)
    return {
        "text": result["text"],
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ]
    }
```

## 6. 成本优化

### 6.1 缓存策略

| 数据 | 缓存方式 | 失效条件 |
|---|---|---|
| Gemini 文件上传 | 路径+大小哈希 | 文件修改 |
| CLIP 向量 | SQLite blob 字段 | 素材删除 |
| Whisper 转录 | JSON 文件 | 音轨未变 |
| 场景分析结果 | SQLite JSON 字段 | 素材重新分析 |

### 6.2 降频策略

- 转码后用低分辨率版本做分析（480p），减少 API token
- CLIP 向量化用抽帧而非逐帧（每 2 秒一帧）
- Whisper 用 base 模型而非 large（速度快 5 倍，旅行 Vlog 足够）
