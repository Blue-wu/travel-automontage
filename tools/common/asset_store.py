"""素材库持久化层 — SQLite 存储 Asset 元数据 + 向量"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from tools.common.config import ASSETS_DB_PATH

logger = logging.getLogger(__name__)
from tools.common.models import Asset, Scene


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id      TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL UNIQUE,
    normalized_path TEXT,
    destination   TEXT,
    metadata_json TEXT NOT NULL,
    scenes_json   TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assets_destination ON assets(destination);

CREATE TABLE IF NOT EXISTS scene_vectors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL,
    scene_index INTEGER NOT NULL,
    summary     TEXT,
    tags_json   TEXT,
    embedding   BLOB,  -- numpy float32 数组的 bytes
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vectors_asset ON scene_vectors(asset_id);
"""


class AssetStore:
    """素材库 — SQLite 后端，支持元数据检索和向量检索"""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else ASSETS_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate(conn)

    def _migrate(self, conn):
        """数据库迁移：按需添加新字段"""
        # 检查 video_embedding 字段是否存在
        cursor = conn.execute("PRAGMA table_info(scene_vectors)")
        columns = {row[1] for row in cursor.fetchall()}
        if "video_embedding" not in columns:
            conn.execute("ALTER TABLE scene_vectors ADD COLUMN video_embedding BLOB")
            logger.info("迁移完成：scene_vectors 添加 video_embedding 字段")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save(self, asset: Asset) -> None:
        """保存或更新素材资产"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assets (asset_id, source_path, normalized_path, destination, metadata_json, scenes_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.asset_id,
                    asset.source_path,
                    asset.normalized_path,
                    asset.destination,
                    asset.metadata.model_dump_json(),
                    json.dumps([s.model_dump() for s in asset.scenes], ensure_ascii=False),
                ),
            )

            # 删除旧的场景向量，重新插入
            conn.execute("DELETE FROM scene_vectors WHERE asset_id = ?", (asset.asset_id,))
            for i, scene in enumerate(asset.scenes):
                import numpy as np

                embedding_blob = None
                if scene.embedding:
                    embedding_blob = np.array(scene.embedding, dtype=np.float32).tobytes()

                video_embedding_blob = None
                if hasattr(scene, 'video_embedding') and scene.video_embedding:
                    video_embedding_blob = np.array(scene.video_embedding, dtype=np.float32).tobytes()

                conn.execute(
                    """
                    INSERT INTO scene_vectors (asset_id, scene_index, summary, tags_json, embedding, video_embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset.asset_id,
                        i,
                        scene.summary,
                        json.dumps(scene.visual_tags, ensure_ascii=False),
                        embedding_blob,
                        video_embedding_blob,
                    ),
                )

    def get(self, asset_id: str) -> Asset | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
            if not row:
                return None
            return self._row_to_asset(row)

    def get_by_path(self, source_path: str) -> Asset | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE source_path = ?", (source_path,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_asset(row)

    def list_all(self) -> Iterator[Asset]:
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM assets ORDER BY created_at DESC"):
                yield self._row_to_asset(row)

    def filter_by_destination(self, destination: str) -> list[Asset]:
        """按目的地过滤素材"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE destination LIKE ? ORDER BY created_at DESC",
                (f"%{destination}%",),
            ).fetchall()
            return [self._row_to_asset(r) for r in rows]

    def get_all_embeddings(self, destination: str | None = None) -> list[tuple[str, int, str, list[float]]]:
        """获取所有场景向量（用于语义检索）

        Returns:
            list of (asset_id, scene_index, summary, embedding)
        """
        import numpy as np

        with self._connect() as conn:
            if destination:
                rows = conn.execute(
                    """
                    SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, a.source_path
                    FROM scene_vectors sv
                    JOIN assets a ON sv.asset_id = a.asset_id
                    WHERE a.destination LIKE ?
                    """,
                    (f"%{destination}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, a.source_path
                    FROM scene_vectors sv
                    JOIN assets a ON sv.asset_id = a.asset_id
                    """
                ).fetchall()

        results = []
        for r in rows:
            if r["embedding"] is None:
                continue
            emb = np.frombuffer(r["embedding"], dtype=np.float32).tolist()
            results.append((r["asset_id"], r["scene_index"], r["summary"], emb))
        return results

    def get_all_embeddings_v2(self, destination: str | None = None) -> list[tuple[str, int, str, list[float], list[float] | None]]:
        """获取所有场景向量（V2：CLIP + VideoMAE 双向量）

        Returns:
            list of (asset_id, scene_index, summary, clip_embedding, video_embedding)
        """
        import numpy as np

        with self._connect() as conn:
            # 先检查 video_embedding 字段是否存在
            cursor = conn.execute("PRAGMA table_info(scene_vectors)")
            columns = {row[1] for row in cursor.fetchall()}
            has_video = "video_embedding" in columns

            if destination:
                if has_video:
                    rows = conn.execute(
                        """
                        SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, sv.video_embedding, a.source_path
                        FROM scene_vectors sv
                        JOIN assets a ON sv.asset_id = a.asset_id
                        WHERE a.destination LIKE ?
                        """,
                        (f"%{destination}%",),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, a.source_path
                        FROM scene_vectors sv
                        JOIN assets a ON sv.asset_id = a.asset_id
                        WHERE a.destination LIKE ?
                        """,
                        (f"%{destination}%",),
                    ).fetchall()
            else:
                if has_video:
                    rows = conn.execute(
                        """
                        SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, sv.video_embedding, a.source_path
                        FROM scene_vectors sv
                        JOIN assets a ON sv.asset_id = a.asset_id
                        """
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT sv.asset_id, sv.scene_index, sv.summary, sv.embedding, a.source_path
                        FROM scene_vectors sv
                        JOIN assets a ON sv.asset_id = a.asset_id
                        """
                    ).fetchall()

        results = []
        for r in rows:
            if r["embedding"] is None:
                continue
            clip_emb = np.frombuffer(r["embedding"], dtype=np.float32).tolist()

            video_emb = None
            if has_video and r["video_embedding"] is not None:
                video_emb = np.frombuffer(r["video_embedding"], dtype=np.float32).tolist()

            results.append((r["asset_id"], r["scene_index"], r["summary"], clip_emb, video_emb))
        return results

    def delete(self, asset_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM scene_vectors WHERE asset_id = ?", (asset_id,))
            conn.execute("DELETE FROM assets WHERE asset_id = ?", (asset_id,))

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    def _row_to_asset(self, row: sqlite3.Row) -> Asset:
        from tools.common.models import AssetMetadata

        scenes_data = json.loads(row["scenes_json"])
        scenes = [Scene(**s) for s in scenes_data]
        metadata = AssetMetadata.model_validate_json(row["metadata_json"])

        return Asset(
            asset_id=row["asset_id"],
            source_path=row["source_path"],
            normalized_path=row["normalized_path"] or "",
            destination=row["destination"] or "",
            scenes=scenes,
            metadata=metadata,
        )
