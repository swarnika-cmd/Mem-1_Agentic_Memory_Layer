from typing import List, Dict, Any, Optional
import json
import logging
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore
from mem1.memory import SemanticMemoryManager

logger = logging.getLogger("mem1.consolidation")

class ConsolidationEngine:
    """Consolidates episodic memories into semantic knowledge graphs and removes redundancy."""

    def __init__(
        self, 
        db: BaseRelationalDB, 
        vector_store: BaseVectorStore, 
        graph_store: BaseGraphStore,
        semantic_manager: Optional[SemanticMemoryManager] = None
    ):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.semantic_manager = semantic_manager or SemanticMemoryManager(db, vector_store, graph_store)

    def consolidate_episodic_memories(self) -> List[Dict[str, Any]]:
        """Scans SQLite database for unconsolidated episodic memories and merges them into semantic facts."""
        # 1. Fetch all raw memories
        # We check sqlite_db's helper to get all memories
        if not hasattr(self.db, "list_all_memories"):
            return []

        all_memories = self.db.list_all_memories()
        
        # 2. Filter for unconsolidated episodic memories
        unconsolidated = []
        for mem in all_memories:
            if mem["memory_type"] == "episodic":
                meta = mem["metadata"]
                if not meta.get("consolidated", False):
                    unconsolidated.append(mem)

        if len(unconsolidated) < 2:
            # We need at least some episodic events to perform a consolidation pass
            return []

        new_semantic_facts = []
        
        # 3. Simple clustering/grouping: for the MVP, we cluster episodic memories by shared keywords
        # in a production system, this sends batches of dialogues to an LLM
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for mem in unconsolidated:
            content = mem["content"].lower()
            # Simple heuristic matching for entity/topic grouping
            detected_topic = "general"
            detected_entity = "User"
            
            if "sqlite" in content or "database" in content or "db" in content:
                detected_topic = "database_choice"
                detected_entity = "SQLite"
            elif "react" in content or "frontend" in content or "dashboard" in content:
                detected_topic = "frontend_ui"
                detected_entity = "React"
            elif "fastapi" in content or "backend" in content or "api" in content:
                detected_topic = "backend_framework"
                detected_entity = "FastAPI"

            key = f"{detected_entity}:{detected_topic}"
            if key not in groups:
                groups[key] = []
            groups[key].append(mem)

        # 4. Process each group and synthesize semantic knowledge
        for key, mem_list in groups.items():
            if len(mem_list) < 2:
                continue  # Skip groups with single memories (insufficient context to consolidate)

            entity, topic = key.split(":")
            
            # Synthesize semantic fact text
            synthesized_fact = f"User preference consolidated: Agent learned that for {entity} related tasks ({topic}), " \
                               f"multiple interactions occurred. Consolidated details: "
            
            summaries = []
            for m in mem_list:
                summaries.append(m["content"])
            
            synthesized_fact += " | ".join(summaries)

            # Store the new semantic memory
            semantic_node = self.semantic_manager.create_semantic_memory(
                content=synthesized_fact,
                entity_name=entity,
                topic=topic,
                metadata={"consolidated_from": [m["id"] for m in mem_list]}
            )

            # 5. Graph relationships: link episodic source events to the new semantic concept node
            # Event -> derived_from -> Concept
            for m in mem_list:
                self.graph_store.add_edge(m["id"], semantic_node.id, "derived_from")
                
                # Mark episodic memory as consolidated in SQLite
                meta = m["metadata"]
                meta["consolidated"] = True
                meta["consolidation_target"] = semantic_node.id
                self.db.update_raw_memory(m["id"], m["content"], meta)

            new_semantic_facts.append({
                "semantic_id": semantic_node.id,
                "fact": synthesized_fact,
                "sources": [m["id"] for m in mem_list]
            })

        return new_semantic_facts
