import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from mem1.database.base import BaseRelationalDB

class SQLiteDB(BaseRelationalDB):
    """SQLite implementation for local raw memories and logs storage."""

    def __init__(self, db_path: str = "mem1_local.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.connect()

    def connect(self) -> None:
        """Establish connection and initialize tables if they don't exist."""
        # Ensure parent directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables for raw memories and logs."""
        cursor = self.conn.cursor()
        
        # Raw memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Retrieval logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                retrieved_ids TEXT NOT NULL,
                feedback REAL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def close(self) -> None:
        """Close connection to the database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def store_raw_memory(self, memory_id: str, memory_type: str, content: str, metadata: Dict[str, Any]) -> None:
        """Store or replace a raw memory entry with metadata."""
        now = datetime.utcnow().isoformat()
        metadata_str = json.dumps(metadata)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO raw_memories (id, memory_type, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, memory_type, content, metadata_str, now, now))
        self.conn.commit()

    def get_raw_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve raw memory entry by its ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM raw_memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "id": row["id"],
            "memory_type": row["memory_type"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def update_raw_memory(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Update an existing raw memory entry's content and metadata."""
        now = datetime.utcnow().isoformat()
        metadata_str = json.dumps(metadata)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE raw_memories
            SET content = ?, metadata = ?, updated_at = ?
            WHERE id = ?
        """, (content, metadata_str, now, memory_id))
        self.conn.commit()

    def delete_raw_memory(self, memory_id: str) -> None:
        """Delete a raw memory entry by ID."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM raw_memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def log_retrieval(self, query: str, retrieved_ids: List[str], feedback: Optional[float] = None) -> None:
        """Log retrieval operations for debugging and evaluation."""
        now = datetime.utcnow().isoformat()
        ids_str = json.dumps(retrieved_ids)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO retrieval_logs (query, retrieved_ids, feedback, timestamp)
            VALUES (?, ?, ?, ?)
        """, (query, ids_str, feedback, now))
        self.conn.commit()

    def list_all_memories(self) -> List[Dict[str, Any]]:
        """Helper to dump all memories for dashboard display."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM raw_memories ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]
