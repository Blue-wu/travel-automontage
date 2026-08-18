#!/usr/bin/env python3
"""向量库一致性诊断 — 检查 scene_vectors 里的向量是否来自同一个模型空间

背景：入库走 hybrid 模式时存的是 open_clip 512 维向量，
而检索时 model_client.embed_text() 优先用 qwen3-vl-embedding（1024 维）。
两侧独立决定模型，DB 里也没记录每条向量的来源，可能出现：
  - 维度不匹配 → np.dot 直接抛异常
  - 同一次入库中途降级 → 库内混合空间
  - 两侧都回退英文 CLIP → 中文 query 失效

用法：
    python diagnose_vectors.py
    python diagnose_vectors.py --db path/to/assets.sqlite3
    python diagnose_vectors.py --check-query      # 额外实测检索侧维度（需要环境依赖）

不查询时无第三方依赖。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

FLOAT32 = 4

# 维度 → 可能的模型来源
DIM_HINTS = {
    512:  "open_clip ViT-B-32 (英文, laion2b)  ← clip_classifier 默认",
    768:  "open_clip ViT-L-14 / VideoMAE-base",
    1024: "qwen3-vl-embedding (中文原生)      ← config 默认 embedding_dim",
    1152: "SigLIP so400m",
    1280: "open_clip ViT-H-14",
    1536: "qwen3-vl-embedding (1536 档)",
    2048: "qwen2.5-vl-embedding",
    2560: "qwen3-vl-embedding (默认 2560 档)",
}


def hint(dim):
    return DIM_HINTS.get(dim, "未知模型")


def find_db(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"✗ 找不到数据库: {p}")
        return p
    for c in [Path("data/assets_db/assets.sqlite3"),
              Path(__file__).parent / "data/assets_db/assets.sqlite3"]:
        if c.exists():
            return c
    sys.exit("✗ 找不到 data/assets_db/assets.sqlite3，用 --db 指定")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--check-query", action="store_true",
                    help="实测 model_client.embed_text() 返回的维度")
    args = ap.parse_args()

    db = find_db(args.db)
    print(f"数据库: {db.resolve()}")
    print(f"大小:   {db.stat().st_size / 1024:.0f} KB\n")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(scene_vectors)")}
    if not cols:
        sys.exit("✗ 表 scene_vectors 不存在")
    has_video = "video_embedding" in cols

    n_assets = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    n_vec = conn.execute("SELECT COUNT(*) FROM scene_vectors").fetchone()[0]
    print(f"素材数: {n_assets}   场景向量数: {n_vec}")
    print(f"video_embedding 列: {'有' if has_video else '无'}\n")
    if n_vec == 0:
        sys.exit("素材库为空，无可诊断内容。")

    vsel = "sv.video_embedding," if has_video else ""
    rows = list(conn.execute(f"""
        SELECT sv.id, sv.asset_id, sv.scene_index, sv.embedding, {vsel}
               a.source_path, a.created_at
        FROM scene_vectors sv
        LEFT JOIN assets a ON sv.asset_id = a.asset_id
        ORDER BY sv.id
    """))

    dims = Counter()
    vdims = Counter()
    nulls = 0
    per_asset = defaultdict(set)
    order = []          # (created_at, asset_id, path, dim) 保持入库顺序

    for r in rows:
        blob = r["embedding"]
        if blob is None:
            nulls += 1
            d = None
        else:
            d = len(blob) // FLOAT32
            dims[d] += 1
            per_asset[r["asset_id"]].add(d)
        order.append((r["id"], r["created_at"], r["asset_id"], r["source_path"] or "?", d))

        if has_video and r["video_embedding"] is not None:
            vdims[len(r["video_embedding"]) // FLOAT32] += 1

    # ── 1. 主向量维度分布 ──
    print("=" * 68)
    print("【1】主向量 (embedding) 维度分布")
    print("=" * 68)
    if nulls:
        print(f"  NULL（无向量）      : {nulls} 条  ← 这些场景检索时被跳过")
    for d, c in dims.most_common():
        pct = c / max(1, sum(dims.values())) * 100
        print(f"  {d:>5} 维  {c:>6} 条 ({pct:5.1f}%)   {hint(d)}")

    if has_video and vdims:
        print("\n【1b】video_embedding 维度分布")
        for d, c in vdims.most_common():
            print(f"  {d:>5} 维  {c:>6} 条   {hint(d)}")

    # ── 2. 单个素材内部是否混合 ──
    mixed_assets = {a: ds for a, ds in per_asset.items() if len(ds) > 1}
    print("\n" + "=" * 68)
    print("【2】单个素材内部是否混合空间")
    print("=" * 68)
    if mixed_assets:
        print(f"  ✗ {len(mixed_assets)} 个素材内部存在多种维度（同一次分析中途降级）")
        for a, ds in list(mixed_assets.items())[:5]:
            print(f"      {a[:16]}  维度={sorted(ds)}")
    else:
        print("  ✓ 每个素材内部维度一致")

    # ── 3. 入库过程中是否发生过降级 ──
    print("\n" + "=" * 68)
    print("【3】入库时间线上的维度切换点")
    print("=" * 68)
    # 按 scene_vectors.id 自增主键排序 = 真实写入顺序
    # （assets.created_at 只精确到秒，同批入库会大量并列，不能用来定序）
    switches = []
    prev = None
    for vid, ts, aid, path, d in order:
        if d is None:
            continue
        if prev is not None and d != prev:
            switches.append((vid, ts, Path(path).name, prev, d))
        prev = d
    if switches:
        print(f"  ✗ 检测到 {len(switches)} 次维度切换（写入过程中换过模型）")
        for vid, ts, name, a, b in switches[:8]:
            print(f"      #{vid:<6} {ts}  {name[:30]:<30} {a} → {b}")
        if len(switches) > 8:
            print(f"      ... 另有 {len(switches) - 8} 次")
        if len(switches) == 1:
            vid = switches[0][0]
            print(f"\n      只切换 1 次 → 典型的「跑到一半 API 挂了自动降级」")
            print(f"      #{vid} 之前的向量可保留，之后的需要重算")
    else:
        print("  ✓ 全程维度稳定，未发生中途降级")

    # ── 4. 检索侧实测 ──
    query_dim = None
    if args.check_query:
        print("\n" + "=" * 68)
        print("【4】检索侧 embed_text() 实测")
        print("=" * 68)
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from tools.common.model_client import ModelClient
            vec = ModelClient().embed_text("航拍俯瞰草原 牛羊 雪山远景")
            if vec is None:
                print("  ✗ embed_text 返回 None —— 检索会走关键词回退")
            else:
                query_dim = len(vec)
                print(f"  query 向量维度: {query_dim}   {hint(query_dim)}")
        except Exception as e:
            print(f"  ! 实测失败（缺依赖或无 API Key）: {e}")

    # ── 5. 结论 ──
    print("\n" + "=" * 68)
    print("【结论】")
    print("=" * 68)
    uniq = sorted(dims)
    if len(uniq) > 1:
        print(f"  ✗✗ 严重：库内存在 {len(uniq)} 种向量空间 {uniq}")
        print("     不同空间的向量互相比较余弦相似度没有意义，")
        print("     且与 query 维度不同的那部分会让 np.dot 直接抛异常。")
        print("     → 必须全量重新向量化，并固定单一模型。")
    elif uniq:
        d = uniq[0]
        print(f"  ✓ 库内向量空间统一：{d} 维（{hint(d)}）")
        if d == 512:
            print("  ✗ 但这是英文 open_clip —— 中文 query 语义对不上，")
            print("     这是检索'不贴切'的首要原因。")
    if query_dim is not None and uniq:
        if query_dim not in uniq:
            print(f"  ✗✗ 检索侧 {query_dim} 维 ≠ 库内 {uniq} 维 —— 检索必然报错或结果无意义！")
        else:
            print(f"  ✓ 检索侧维度 {query_dim} 与库内一致")
    if nulls:
        print(f"  ! {nulls} 条场景没有向量，永远检索不到")

    conn.close()


if __name__ == "__main__":
    main()
