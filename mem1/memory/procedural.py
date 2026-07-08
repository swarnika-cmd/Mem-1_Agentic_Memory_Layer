import uuid
import json
from typing import Dict, Any, List, Optional
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore
from mem1.memory.base import MemoryNode

class ProceduralMemoryManager:
    """Handles storing, indexing, and graph-linking procedural memory (workflows, task histories, and plans)."""

    def __init__(self, db: BaseRelationalDB, vector_store: BaseVectorStore, graph_store: BaseGraphStore):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store

    def create_procedural_memory(
        self, 
        content: str, 
        task_name: str, 
        steps: List[str], 
        outcome: str = "success",  # "success" or "failure"
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryNode:
        """Create, index, and graph-link a procedural memory entry."""
        meta = metadata.copy() if metadata else {}
        meta["task_name"] = task_name
        meta["steps"] = steps
        meta["outcome"] = outcome
        meta["type"] = "procedural"

        memory_id = f"pro_{uuid.uuid4().hex[:12]}"

        # 1. Store raw memory in relational database
        self.db.store_raw_memory(memory_id, "procedural", content, meta)

        # 2. Add to vector store for workflow lookup
        self.vector_store.add_vector(memory_id, f"Task: {task_name}. Steps: {' -> '.join(steps)}. Outcome: {outcome}. Context: {content}")

        # 3. Add to Graph Store as a "procedure" node
        self.graph_store.add_node(memory_id, "procedure", {
            "content": content,
            "task_name": task_name,
            "steps": json.dumps(steps),
            "outcome": outcome
        })

        # 4. Task Node Linking: link this execution to a central Task Name entity
        task_node_id = f"task_{task_name.lower().replace(' ', '_')}"
        if not self.graph_store.get_node(task_node_id):
            self.graph_store.add_node(task_node_id, "entity", {
                "name": task_name,
                "description": f"Workflow task for {task_name}"
            })
        
        # Link Task -> has_execution -> Procedure Execution
        self.graph_store.add_edge(task_node_id, memory_id, "has_execution")

        return MemoryNode(
            id=memory_id,
            type="procedural",
            content=content,
            metadata=meta
        )
