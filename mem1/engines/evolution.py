from typing import Dict, Any, Optional, List
from datetime import datetime
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore

class EvolutionEngine:
    """Handles updating memory nodes, tracking version history, and managing conflicting memories."""

    def __init__(self, db: BaseRelationalDB, vector_store: BaseVectorStore, graph_store: BaseGraphStore):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store

    def update_memory(
        self, 
        node_id: str, 
        new_content: str, 
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Updates a memory's content, increases its version, and records history logs."""
        raw_mem = self.db.get_raw_memory(node_id)
        if not raw_mem:
            raise ValueError(f"Memory node with ID {node_id} does not exist.")

        old_content = raw_mem["content"]
        old_metadata = raw_mem["metadata"]
        old_updated_at = raw_mem["updated_at"]

        # Track version number
        current_version = old_metadata.get("version", 1)
        new_version = current_version + 1

        # Append to historical changelog list
        history = old_metadata.get("history", [])
        history.append({
            "version": current_version,
            "content": old_content,
            "updated_at": old_updated_at,
            "reason": additional_metadata.get("update_reason", "Manual update") if additional_metadata else "Manual update"
        })

        # Merge metadata
        new_metadata = old_metadata.copy()
        if additional_metadata:
            new_metadata.update(additional_metadata)
        
        new_metadata["version"] = new_version
        new_metadata["history"] = history
        new_metadata["last_updated"] = datetime.utcnow().isoformat()

        # 1. Update Relational Database
        self.db.update_raw_memory(node_id, new_content, new_metadata)

        # 2. Update Vector Store
        self.vector_store.remove_vector(node_id)
        self.vector_store.add_vector(node_id, new_content)

        # 3. Update Graph Store Node Properties
        self.graph_store.update_node(node_id, {
            "content": new_content,
            "version": new_version
        })

        # 4. Check for contradiction heuristics
        # Simple heuristic: if positive state turns negative or vice versa
        self._detect_and_link_contradictions(node_id, old_content, new_content)

        return {
            "node_id": node_id,
            "old_content": old_content,
            "new_content": new_content,
            "version": new_version
        }

    def _detect_and_link_contradictions(self, node_id: str, old_content: str, new_content: str) -> None:
        """Helper to create a 'contradicts' self-relationship edge in the graph if a reversal is detected."""
        old_clean = old_content.lower()
        new_clean = new_content.lower()

        contradiction_keywords = [
            ("instead of", "prefer"),
            ("change to", "no longer"),
            ("switch", "replace")
        ]

        is_contradicting = False
        # Heuristic 1: keyword check
        for old_kw, new_kw in contradiction_keywords:
            if old_kw in old_clean or new_kw in new_clean:
                is_contradicting = True
                break

        # Heuristic 2: direct negative transition
        negatives = ["not", "dont", "don't", "never", "disable", "stop"]
        for neg in negatives:
            if (neg in new_clean and neg not in old_clean) or (neg in old_clean and neg not in new_clean):
                # If one has "never use SQLite" and other has "use SQLite"
                # Check if they share subject nouns
                subjects = ["sqlite", "postgres", "fastapi", "react"]
                for subj in subjects:
                    if subj in old_clean and subj in new_clean:
                        is_contradicting = True
                        break

        if is_contradicting:
            # Add self-directed relation: node -> contradicts -> node (representing state transition friction)
            self.graph_store.add_edge(node_id, node_id, "contradicts", weight=0.8)
