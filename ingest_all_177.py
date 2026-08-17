"""全量入库脚本：177个新疆旅行视频混合模式入库

特性：
- 使用混合策略（本地 CLIP + qwen 精选增强）
- 支持断点续传（skip_existing=True）
- 实时进度条和预计剩余时间
- 异常不中断，失败视频记录在日志

用法：
  python ingest_all_177.py
"""

import logging
import os
import sys
import time
import glob
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/ingest_all.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ingest_all")

# ========== 配置 ==========
FOOTAGE_DIR = r"C:\Users\blue\Downloads\0527"
DESTINATION = "新疆"
DB_PATH = "data/assets_db/xinjiang_177.sqlite3"  # 用独立 DB，不影响测试
SKIP_TRANSCODE = True
HYBRID_MODE = True
# ==========================

from tools.common.config import get_settings
settings = get_settings()

if not settings.dashscope.api_key:
    print("=" * 60)
    print("ERROR: DASHSCOPE_API_KEY 未设置！")
    print("=" * 60)
    sys.exit(1)

print(f"VL 模型: {settings.dashscope.vl_model}")
print(f"目标目录: {FOOTAGE_DIR}")
print(f"目的地标签: {DESTINATION}")
print(f"数据库: {DB_PATH}")
print(f"模式: 混合模式={HYBRID_MODE}, 跳过转码={SKIP_TRANSCODE}")

# 发现所有视频
all_videos = sorted(glob.glob(os.path.join(FOOTAGE_DIR, "*.MP4")))
total_count = len(all_videos)
print(f"\n发现视频: {total_count} 个")

if total_count == 0:
    print("没有找到视频！")
    sys.exit(1)

from tools.common.asset_store import AssetStore
from tools.common.model_client import ModelClient
from tools.ingest.travel_asset_ingestor import TravelAssetIngestor

# 初始化
store = AssetStore(db_path=DB_PATH)
ingestor = TravelAssetIngestor(asset_store=store, model_client=ModelClient())

# 检查已完成数量（断点续传）
already_done = 0
for v in all_videos:
    if store.get_by_path(v) is not None:
        already_done += 1

print(f"已入库: {already_done}/{total_count}，待处理: {total_count - already_done}")
if already_done > 0:
    print("  (断点续传模式，跳过已入库视频)")

print("\n" + "=" * 60)
print("开始入库... 按 Ctrl+C 可安全中断，下次自动续传")
print("=" * 60)

# 进度统计
success_count = already_done
fail_count = 0
fail_list = []
total_scenes = 0
total_qwen_enhanced = 0
start_time = time.time()

for idx, v in enumerate(all_videos):
    video_name = os.path.basename(v)
    progress_pct = (idx) / total_count * 100

    # 跳过已入库
    if store.get_by_path(v) is not None:
        print(f"[{idx+1:>3}/{total_count}] 跳过 {video_name} (已存在)")
        continue

    # 进度行
    elapsed = time.time() - start_time
    if idx > already_done:
        per_video = elapsed / (idx - already_done)
        remaining = (total_count - idx) * per_video
        eta_str = f"剩余约 {remaining/60:.0f} 分钟"
    else:
        eta_str = "计算中..."

    print(f"\n[{idx+1:>3}/{total_count}] ({progress_pct:5.1f}%) {video_name}  | {eta_str}")
    video_start = time.time()

    try:
        asset = ingestor._ingest_single(
            Path(v),
            skip_transcode=SKIP_TRANSCODE,
            override_destination=DESTINATION,
            hybrid_mode=HYBRID_MODE,
        )
        if asset:
            store.save(asset)
            elapsed_v = time.time() - video_start
            n_scenes = len(asset.scenes)
            total_scenes += n_scenes
            n_enhanced = sum(1 for s in asset.scenes if "新疆" in s.summary or "，" in s.summary)
            total_qwen_enhanced += n_enhanced
            success_count += 1
            avg_q = sum(s.quality for s in asset.scenes) / max(1, n_scenes)
            print(f"  ✅ 成功: {n_scenes} 场景, 平均质量={avg_q:.2f}, qwen增强={n_enhanced}, 耗时={elapsed_v:.0f}s")
        else:
            fail_count += 1
            fail_list.append(video_name)
            print(f"  ❌ 失败: 返回空")
    except Exception as e:
        fail_count += 1
        fail_list.append(video_name)
        logger.error(f"入库异常 {video_name}: {e}", exc_info=True)
        print(f"  ❌ 异常: {e}")
        continue

# ==========================
# 完成总结
# ==========================
print("\n" + "=" * 60)
print("全量入库完成！")
print("=" * 60)

total_elapsed = time.time() - start_time
hours = int(total_elapsed // 3600)
mins = int((total_elapsed % 3600) // 60)

print(f"总耗时: {hours}小时{mins}分钟")
print(f"成功: {success_count}/{total_count}")
print(f"失败: {fail_count}")
if fail_list:
    print(f"失败列表: {fail_list}")
print(f"场景总数: {total_scenes} (平均 {total_scenes/max(1,success_count):.1f} 场景/视频)")
print(f"qwen 增强描述: {total_qwen_enhanced} 个场景")
print(f"Token 估算: {total_qwen_enhanced * 1000:,} (~{total_qwen_enhanced*1000/10000:.1f}万)")
print(f"数据库: {DB_PATH}")

# 日志路径
abs_db = os.path.abspath(DB_PATH)
log_path = os.path.abspath("data/ingest_all.log")
print(f"\n日志文件: {log_path}")

# 简单验证：显示一下类别分布
print("\n--- 场景类别分布 ---")
try:
    cats = {}
    for asset in store.list_all():
        for s in asset.scenes:
            c = s.scene_category or "other"
            cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * int(n / max(cats.values()) * 30)
        print(f"  {c:<16} {n:>4} {bar}")
except Exception as e:
    print(f"(统计失败: {e})")
