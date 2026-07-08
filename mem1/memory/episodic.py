import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore
from mem1.memory.base import MemoryNode

class EpisodicMemoryManager:
    """Handles storing, updating, and graph-linking episodic (temporal event) memories."""

    def __init__(self, db: BaseRelationalDB, vector_store: BaseVectorStore, graph_store: BaseGraphStore):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store

    def create_episodic_memory(
        self, 
        content: str, 
        session_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryNode:
        """Create, index, and graph-link an episodic memory."""
        meta = metadata.copy() if metadata else {}
        meta["session_id"] = session_id
        meta["timestamp"] = meta.get("timestamp", datetime.utcnow().isoformat())
        meta["type"] = "episodic"

        memory_id = f"epi_{uuid.uuid4().hex[:12]}"
        
        # 1. Store raw memory in relational SQLite database
        self.db.store_raw_memory(memory_id, "episodic", content, meta)

        # 2. Add to vector store for semantic retrieval
        self.vector_store.add_vector(memory_id, content)

        # 3. Add to Graph Store as an "event" node
        self.graph_store.add_node(memory_id, "event", {
            "content": content,
            "session_id": session_id,
            "timestamp": meta["timestamp"]
        })

        # 4. Link to the Session Entity node
        session_node_id = f"sess_{session_id}"
        if not self.graph_store.get_node(session_node_id):
            self.graph_store.add_node(session_node_id, "entity", {
                "name": f"Session {session_id}",
                "session_id": session_id
            })
        self.graph_store.add_edge(session_node_id, memory_id, "contained_event")

        # 5. Temporal Linking: Link to the last event in the session if available
        # We query the relational database to find the last episodic memory for this session
        cursor = getattr(self.db, "conn", None)
        if cursor:
            try:
                db_cursor = cursor.cursor()
                db_cursor.execute(
                    "SELECT id FROM raw_memories WHERE memory_type = 'episodic' AND id != ? ORDER BY created_at DESC LIMIT 1",
                    (memory_id,)
                )
                row = db_cursor.fetchone()
                if row:
                    last_event_id = row["id"]
                    # Edge: last_event -> followed_by -> current_event
                    self.graph_store.add_edge(last_event_id, memory_id, "followed_by")
            except Exception:
                pass  # Fallback if DB doesn't support the schema directly

        return MemoryNode(
            id=memory_id,
            type="episodic",
            content=content,
            metadata=meta,
            created_at=meta["timestamp"],
            updated_at=meta["timestamp"]
        )
