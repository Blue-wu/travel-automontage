"""混合策略测试：本地 CLIP 切分+分类+向量化 + qwen 精选增强

成本对比：
  - 旧方案（整视频上传 qwen）：~15万 token/视频
  - 混合方案（CLIP + qwen 图片增强）：~1000 token/精彩场景

用法：
  python test_qwen_pipeline.py
"""

import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# 检查 API Key
from tools.common.config import get_settings
settings = get_settings()

print(f"DashScope API Key: {settings.dashscope.api_key[:6]}...")
print(f"VL 模型: {settings.dashscope.vl_model}")

# 用独立测试数据库
TEST_DB = "data/assets_db/test_hybrid.sqlite3"
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

from tools.common.asset_store import AssetStore
from tools.common.model_client import ModelClient
from tools.retrieve.semantic_asset_retriever import SemanticAssetRetriever
from tools.ingest.travel_asset_ingestor import TravelAssetIngestor

footage_dir = r"C:\Users\blue\Downloads\0527"

import glob
all_videos = sorted(glob.glob(os.path.join(footage_dir, "*.MP4")))
test_videos = all_videos[:5]  # 先测 5 个

print("=" * 60)
print("混合策略测试：本地 CLIP + qwen 精选增强")
print("=" * 60)
print(f"\n测试视频: {len(test_videos)} 个")
print(f"策略: PySceneDetect 切分 → CLIP 分类+向量化 → qwen 增强精彩场景描述")

# Step 1: 入库
print("\n--- Step 1: 入库（混合模式）---")
store = AssetStore(db_path=TEST_DB)
ingestor = TravelAssetIngestor(asset_store=store, model_client=ModelClient())

total_start = time.time()

for v in test_videos:
    print(f"\n入库: {os.path.basename(v)}")
    t0 = time.time()
    try:
        asset = ingestor._ingest_single(
            Path(v),
            skip_transcode=True,
            override_destination="新疆",
            hybrid_mode=True,  # 混合模式
        )
        elapsed = time.time() - t0
        if asset:
            store.save(asset)
            print(f"  场景数: {len(asset.scenes)}  耗时: {elapsed:.1f}s")
            for s in asset.scenes:
                has_emb = "YES" if s.embedding else "NO"
                emb_dim = len(s.embedding) if s.embedding else 0
                print(f"  - {s.start_sec:.1f}s-{s.end_sec:.1f}s [{s.scene_category}] "
                      f"emb={has_emb}({emb_dim}d) q={s.quality:.2f}")
                print(f"    标签: {s.visual_tags[:5]}")
                print(f"    描述: {s.summary[:80]}")
        else:
            print(f"  失败 (耗时 {elapsed:.1f}s)")
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()

total_elapsed = time.time() - total_start
print(f"\n总入库耗时: {total_elapsed:.1f}s (平均 {total_elapsed/len(test_videos):.1f}s/视频)")

# Step 2: 检索测试
print("\n" + "=" * 60)
print("--- Step 2: 检索测试 ---")
print("=" * 60)

retriever = SemanticAssetRetriever(asset_store=store, model_client=ModelClient())

queries = [
    "航拍草原",
    "雪山山脉",
    "湖泊倒影",
    "震撼全景",
    "道路公路",
]

for q in queries:
    print(f"\n查询: '{q}'")
    result = retriever.retrieve(q, destination="新疆", top_k=3)
    print(f"  找到 {result.total_found} 个候选，Top-3:")
    for i, c in enumerate(result.clips):
        print(f"  {i+1}. {os.path.basename(c.source_path)} "
              f"[{c.start_sec:.1f}s-{c.end_sec:.1f}s] "
              f"score={c.score:.3f} [{c.scene_category}]")
        # 打印描述便于人工判断
        if hasattr(c, 'summary') and c.summary:
            print(f"     描述: {c.summary[:60]}")

print(f"\n测试完成。数据库: {TEST_DB}")
print(f"\n成本估算: 5 个视频 × ~2 精彩场景/视频 × ~1000 token/图片 = ~10000 token")
print(f"旧方案成本: 5 个视频 × ~150000 token/视频 = ~750000 token")
print(f"节省: ~98%")
