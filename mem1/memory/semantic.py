import uuid
from typing import Dict, Any, Optional
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore
from mem1.memory.base import MemoryNode

class SemanticMemoryManager:
    """Handles storing, updating, and graph-linking semantic facts, concepts, and entity knowledge."""

    def __init__(self, db: BaseRelationalDB, vector_store: BaseVectorStore, graph_store: BaseGraphStore):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store

    def create_semantic_memory(
        self, 
        content: str, 
        entity_name: Optional[str] = None, 
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryNode:
        """Create, vector-index, and graph-link a semantic memory fact."""
        meta = metadata.copy() if metadata else {}
        meta["entity_name"] = entity_name
        meta["topic"] = topic
        meta["type"] = "semantic"

        memory_id = f"sem_{uuid.uuid4().hex[:12]}"

        # 1. Store raw memory in relational database
        self.db.store_raw_memory(memory_id, "semantic", content, meta)

        # 2. Add to vector store for semantic similarity lookups
        self.vector_store.add_vector(memory_id, content)

        # 3. Add to Graph Store as a "concept" node
        self.graph_store.add_node(memory_id, "concept", {
            "content": content,
            "topic": topic or "general",
            "entity_name": entity_name or "none"
        })

        # 4. Entity Linking: If an entity is specified, link this fact to the Entity node
        if entity_name:
            # Normalize entity node ID
            entity_node_id = f"ent_{entity_name.lower().replace(' ', '_')}"
            if not self.graph_store.get_node(entity_node_id):
                self.graph_store.add_node(entity_node_id, "entity", {
                    "name": entity_name,
                    "description": f"Entity representing {entity_name}"
                })
            
            # Add relationship: Entity -> has_fact -> Concept
            self.graph_store.add_edge(entity_node_id, memory_id, "has_fact")

        return MemoryNode(
            id=memory_id,
            type="semantic",
            content=content,
            metadata=meta
        )
