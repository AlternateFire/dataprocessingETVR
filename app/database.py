"""SQLite session storage for gaze recordings."""

import aiosqlite
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = Path(__file__).parent.parent / "sessions.db"
SESSIONS_DIR = Path(__file__).parent.parent / "sessions_data"
SESSIONS_DIR.mkdir(exist_ok=True)


async def init_db():
    """Initialize database schema."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                csv_path TEXT,
                video_path TEXT,
                video_info TEXT,
                gaze_timeline TEXT,
                session_upload_id TEXT,
                created_at TEXT NOT NULL
            )
        """)
        for col, typ in [("session_upload_id", "TEXT"), ("output_path", "TEXT"), ("status", "TEXT")]:
            try:
                await db.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typ}")
            except Exception:
                pass
        await db.commit()


async def save_session(
    name: str,
    csv_path: Optional[str],
    video_path: Optional[str],
    video_info: Dict,
    gaze_timeline: List[dict],
    session_upload_id: Optional[str] = None,
    output_path: Optional[str] = None,
    status: str = "ready",
) -> int:
    """Save a session to the database. Returns session ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO sessions (name, csv_path, video_path, video_info, gaze_timeline, session_upload_id, output_path, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                csv_path or "",
                video_path or "",
                json.dumps(video_info),
                json.dumps(gaze_timeline),
                session_upload_id or "",
                output_path or "",
                status,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def update_session_output(session_id: int, output_path: str) -> None:
    """Update session with exported video path."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET output_path = ?, status = ? WHERE id = ?",
            (output_path, "ready", session_id),
        )
        await db.commit()


def update_session_output_sync(session_id: int, output_path: str) -> None:
    """Sync version for use from background threads."""
    import sqlite3
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "UPDATE sessions SET output_path = ?, status = ? WHERE id = ?",
            (output_path, "ready", session_id),
        )


async def load_session(session_id: int) -> Optional[Dict]:
    """Load a session by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "csv_path": row["csv_path"] or "",
            "video_path": row["video_path"] or "",
            "video_info": json.loads(row["video_info"] or "{}"),
            "gaze_timeline": json.loads(row["gaze_timeline"] or "[]"),
            "session_upload_id": row["session_upload_id"] or "",
            "output_path": row["output_path"] if "output_path" in row.keys() else "",
            "status": row["status"] if "status" in row.keys() else "ready",
            "created_at": row["created_at"],
        }


async def list_sessions() -> List[Dict]:
    """List all saved sessions."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, csv_path, video_path, output_path, status, created_at FROM sessions ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_session(session_id: int) -> bool:
    """Delete a session."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
        return cursor.rowcount > 0
