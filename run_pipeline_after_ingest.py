"""入库完成后自动运行 pipeline 剪视频

流程：
1. 轮询检查入库进程是否完成（视频数达到 177 或进程消失）
2. 复制 xinjiang_177.sqlite3 → assets.sqlite3（pipeline 默认读取）
3. 运行 travel-vlog pipeline

用法：
  python run_pipeline_after_ingest.py
"""

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/run_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("auto_pipeline")

INGEST_DB = "data/assets_db/xinjiang_177.sqlite3"
DEFAULT_DB = "data/assets_db/assets.sqlite3"
TARGET_VIDEO_COUNT = 177
CHECK_INTERVAL = 60  # 每分钟检查一次


def get_ingest_count() -> int:
    """获取已入库视频数"""
    if not os.path.exists(INGEST_DB):
        return 0
    try:
        conn = sqlite3.connect(INGEST_DB)
        count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def is_ingest_running() -> bool:
    """检查 ingest_all_177.py 进程是否还在运行"""
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=10,
        )
        return "ingest_all_177" in result.stdout
    except Exception:
        # wmic 不可用时，用任务列表
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True, text=True, timeout=10,
            )
            # 如果有 python 进程在跑，保守认为入库还在进行
            return "python.exe" in result.stdout
        except Exception:
            return False


def wait_for_ingest_complete():
    """等待入库完成"""
    log.info("=" * 60)
    log.info("等待入库完成...")
    log.info(f"目标: {TARGET_VIDEO_COUNT} 个视频")
    log.info(f"检查间隔: {CHECK_INTERVAL}s")
    log.info("=" * 60)

    last_count = 0
    stall_count = 0  # 连续无进展次数

    while True:
        count = get_ingest_count()
        running = is_ingest_running()

        if count != last_count:
            pct = count / TARGET_VIDEO_COUNT * 100
            log.info(f"进度: {count}/{TARGET_VIDEO_COUNT} ({pct:.1f}%)")
            last_count = count
            stall_count = 0
        else:
            stall_count += 1

        # 完成 conditions
        if count >= TARGET_VIDEO_COUNT:
            log.info(f"✅ 入库完成！共 {count} 个视频")
            return True

        if not running and count > 0:
            # 进程不在跑了但没到 177
            if stall_count >= 3:
                log.warning(
                    f"⚠️ 入库进程似乎已停止（{count}/{TARGET_VIDEO_COUNT}）"
                )
                if count >= 100:
                    log.info("已入库数量足够，继续剪视频")
                    return True
                else:
                    log.error("入库数量不足，请检查 ingest_all_177.py 是否异常退出")
                    return False

        time.sleep(CHECK_INTERVAL)


def copy_db():
    """复制入库 DB 到 pipeline 默认 DB"""
    log.info(f"复制数据库: {INGEST_DB} → {DEFAULT_DB}")

    # 备份旧 DB（如果存在）
    if os.path.exists(DEFAULT_DB):
        backup = DEFAULT_DB.replace(".sqlite3", "_backup.sqlite3")
        shutil.copy2(DEFAULT_DB, backup)
        log.info(f"  旧 DB 已备份: {backup}")

    shutil.copy2(INGEST_DB, DEFAULT_DB)
    log.info("  ✅ 数据库已就绪")

    # 验证
    conn = sqlite3.connect(DEFAULT_DB)
    count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    scenes = conn.execute("SELECT COUNT(*) FROM scene_vectors").fetchone()[0]
    conn.close()
    log.info(f"  验证: {count} 个视频, {scenes} 个场景向量")


def run_pipeline():
    """运行 travel-vlog pipeline"""
    log.info("=" * 60)
    log.info("开始运行 travel-vlog pipeline")
    log.info("=" * 60)

    cmd = [
        sys.executable, "-m", "tools.cli", "run", "travel-vlog",
        "--destination", "新疆",
    ]
    log.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            timeout=1800,  # 30 分钟超时
        )
        if result.returncode == 0:
            log.info("✅ Pipeline 运行成功！")
            output_video = "data/output/final_新疆.mp4"
            if os.path.exists(output_video):
                size_mb = os.path.getsize(output_video) / (1024 * 1024)
                log.info(f"🎬 成片: {output_video} ({size_mb:.1f} MB)")
            return True
        else:
            log.error(f"❌ Pipeline 失败 (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        log.error("❌ Pipeline 超时（30分钟）")
        return False
    except Exception as e:
        log.error(f"❌ Pipeline 异常: {e}")
        return False


def main():
    # Step 1: 等待入库完成
    ok = wait_for_ingest_complete()
    if not ok:
        log.error("入库未完成，无法继续")
        sys.exit(1)

    # Step 2: 复制 DB
    copy_db()

    # Step 3: 运行 pipeline
    ok = run_pipeline()

    log.info("=" * 60)
    if ok:
        log.info("全部完成！🎬 视频已生成")
    else:
        log.info("Pipeline 运行失败，请查看日志")
    log.info("=" * 60)
    log.info(f"日志文件: {os.path.abspath('data/run_pipeline.log')}")


if __name__ == "__main__":
    main()
