"""Local persistent store for OpenAI-style resources."""
from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Optional


class LocalStore:
    def __init__(self, base_dir: str) -> None:
        os.makedirs(base_dir, exist_ok=True)
        self._db_path = os.path.join(base_dir, "aiproxy_local.db")
        self._files_dir = os.path.join(base_dir, "local_files")
        self._upload_parts_dir = os.path.join(base_dir, "upload_parts")
        os.makedirs(self._files_dir, exist_ok=True)
        os.makedirs(self._upload_parts_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "id TEXT PRIMARY KEY, "
            "type TEXT NOT NULL, "
            "parent_id TEXT, "
            "data TEXT NOT NULL, "
            "created_at INTEGER NOT NULL, "
            "updated_at INTEGER NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS files ("
            "id TEXT PRIMARY KEY, "
            "filename TEXT, "
            "content_type TEXT, "
            "size INTEGER, "
            "path TEXT, "
            "created_at INTEGER NOT NULL, "
            "updated_at INTEGER NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS upload_parts ("
            "id TEXT PRIMARY KEY, "
            "upload_id TEXT NOT NULL, "
            "size INTEGER, "
            "path TEXT, "
            "created_at INTEGER NOT NULL, "
            "updated_at INTEGER NOT NULL"
            ")"
        )
        try:
            self._conn.execute("ALTER TABLE upload_parts ADD COLUMN sha256 TEXT")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def _now(self) -> int:
        return int(time.time())

    def _make_id(self, prefix: str) -> str:
        if not prefix:
            prefix = "obj"
        if prefix.endswith("_") or prefix.endswith("-"):
            return f"{prefix}{uuid.uuid4().hex}"
        return f"{prefix}_{uuid.uuid4().hex}"

    def create_item(
        self,
        item_type: str,
        data: dict,
        parent_id: str | None = None,
        item_id: str | None = None,
        id_prefix: str | None = None,
    ) -> dict:
        item_id = item_id or self._make_id(id_prefix or item_type.replace(".", "_"))
        created_at = self._now()
        item = {"id": item_id, "object": item_type, "created_at": created_at, **data}
        payload = json.dumps(item, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO items (id, type, parent_id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, item_type, parent_id, payload, created_at, created_at),
            )
            self._conn.commit()
        return item

    def list_items(self, item_type: str, parent_id: str | None = None) -> list[dict]:
        query = "SELECT data FROM items WHERE type = ?"
        params: list[Any] = [item_type]
        if parent_id is not None:
            query += " AND parent_id = ?"
            params.append(parent_id)
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_item(self, item_type: str, item_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM items WHERE type = ? AND id = ?",
                (item_type, item_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def update_item(self, item_type: str, item_id: str, updates: dict) -> Optional[dict]:
        existing = self.get_item(item_type, item_id)
        if not existing:
            return None
        updated = dict(existing)
        updated.update({k: v for k, v in updates.items() if k not in ("id", "object", "created_at")})
        payload = json.dumps(updated, ensure_ascii=False)
        updated_at = self._now()
        with self._lock:
            self._conn.execute(
                "UPDATE items SET data = ?, updated_at = ? WHERE type = ? AND id = ?",
                (payload, updated_at, item_type, item_id),
            )
            self._conn.commit()
        return updated

    def delete_item(self, item_type: str, item_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM items WHERE type = ? AND id = ?",
                (item_type, item_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def create_file(self, filename: str, content_type: str, content: bytes) -> dict:
        file_id = self._make_id("file-")
        created_at = self._now()
        path = os.path.join(self._files_dir, file_id)
        with open(path, "wb") as f:
            f.write(content)
        size = len(content)
        with self._lock:
            self._conn.execute(
                "INSERT INTO files (id, filename, content_type, size, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, filename, content_type, size, path, created_at, created_at),
            )
            self._conn.commit()
        return {
            "id": file_id,
            "filename": filename,
            "bytes": size,
            "created_at": created_at,
            "status": "processed",
            "object": "file",
            "expires_at": None,
        }

    def create_file_from_iterator(
        self,
        filename: str,
        content_type: str,
        chunks: Iterable[bytes],
    ) -> dict:
        file_id = self._make_id("file-")
        created_at = self._now()
        path = os.path.join(self._files_dir, file_id)
        size = 0
        with open(path, "wb") as f:
            for chunk in chunks:
                if not chunk:
                    continue
                f.write(chunk)
                size += len(chunk)
        with self._lock:
            self._conn.execute(
                "INSERT INTO files (id, filename, content_type, size, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, filename, content_type, size, path, created_at, created_at),
            )
            self._conn.commit()
        return {
            "id": file_id,
            "filename": filename,
            "bytes": size,
            "created_at": created_at,
            "status": "processed",
            "object": "file",
            "expires_at": None,
        }

    def get_file(self, file_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT filename, content_type, size, path, created_at FROM files WHERE id = ?",
                (file_id,),
            ).fetchone()
        if not row:
            return None
        filename, content_type, size, path, created_at = row
        return {
            "id": file_id,
            "filename": filename,
            "content_type": content_type,
            "bytes": size,
            "path": path,
            "created_at": created_at,
            "status": "processed",
            "object": "file",
            "expires_at": None,
        }

    def delete_file(self, file_id: str) -> bool:
        file_info = self.get_file(file_id)
        with self._lock:
            cur = self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._conn.commit()
        if file_info and os.path.exists(file_info["path"]):
            try:
                os.remove(file_info["path"])
            except OSError:
                pass
        return cur.rowcount > 0

    def create_upload_part(self, upload_id: str, content: bytes) -> dict:
        part_id = self._make_id("upload_part_")
        created_at = self._now()
        path = os.path.join(self._upload_parts_dir, part_id)
        with open(path, "wb") as f:
            f.write(content)
        size = len(content)
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            self._conn.execute(
                "INSERT INTO upload_parts (id, upload_id, size, path, created_at, updated_at, sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (part_id, upload_id, size, path, created_at, created_at, digest),
            )
            self._conn.commit()
        return {
            "id": part_id,
            "object": "upload.part",
            "upload_id": upload_id,
            "bytes": size,
            "created_at": created_at,
            "path": path,
            "sha256": digest,
        }

    def list_upload_parts(self, upload_id: str) -> list[dict]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT id, size, path, created_at, sha256 FROM upload_parts WHERE upload_id = ? ORDER BY created_at ASC",
                    (upload_id,),
                ).fetchall()
                has_hash = True
            except sqlite3.OperationalError:
                rows = self._conn.execute(
                    "SELECT id, size, path, created_at FROM upload_parts WHERE upload_id = ? ORDER BY created_at ASC",
                    (upload_id,),
                ).fetchall()
                has_hash = False
        parts = []
        if has_hash:
            for part_id, size, path, created_at, sha256 in rows:
                parts.append(
                    {
                        "id": part_id,
                        "object": "upload.part",
                        "upload_id": upload_id,
                        "bytes": size or 0,
                        "created_at": created_at,
                        "path": path,
                        "sha256": sha256,
                    }
                )
        else:
            for part_id, size, path, created_at in rows:
                parts.append(
                    {
                        "id": part_id,
                        "object": "upload.part",
                        "upload_id": upload_id,
                        "bytes": size or 0,
                        "created_at": created_at,
                        "path": path,
                        "sha256": None,
                    }
                )
        return parts

    def delete_upload_parts(self, upload_id: str) -> None:
        parts = self.list_upload_parts(upload_id)
        with self._lock:
            self._conn.execute("DELETE FROM upload_parts WHERE upload_id = ?", (upload_id,))
            self._conn.commit()
        for part in parts:
            path = part.get("path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
