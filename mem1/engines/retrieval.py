from typing import List, Dict, Any, Tuple
from datetime import datetime
from mem1.database import BaseRelationalDB, BaseVectorStore, BaseGraphStore
from mem1.engines.scoring import ScoringEngine

class RetrievalEngine:
    """Combines vector similarity and graph traversal to retrieve rich agent context."""

    def __init__(
        self, 
        db: BaseRelationalDB, 
        vector_store: BaseVectorStore, 
        graph_store: BaseGraphStore,
        scoring_engine: Optional[ScoringEngine] = None
    ):
        self.db = db
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.scoring_engine = scoring_engine or ScoringEngine()

    def retrieve_context(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """Perform hybrid (Vector + Graph) retrieval for a query.
        
        1. Find primary node candidates using semantic vector search.
        2. Expand context by traversing connections in the knowledge graph (multi-hop).
        3. Rerank candidates based on decayed importance score & query similarity.
        4. Compile a structured context string for LLM injection.
        """
        # Step 1: Semantic Vector Similarity Search
        vector_results = self.vector_store.search(query, limit=limit * 2)
        if not vector_results:
            return {"memories": [], "formatted_context": "No matching memories found."}

        # Step 2: Retrieve node properties & traverse graph to expand context
        candidate_ids = set()
        for node_id, _ in vector_results:
            candidate_ids.add(node_id)
            
            # Graph Traversal (1-hop neighborhood)
            neighbors = self.graph_store.traverse_graph(node_id, depth=1)
            for n in neighbors:
                candidate_ids.add(n["id"])

        # Step 3: Fetch details and rank candidate memories
        ranked_memories = []
        now_str = datetime.utcnow().isoformat()
        
        for node_id in candidate_ids:
            # Get raw details from sqlite metadata
            raw_mem = self.db.get_raw_memory(node_id)
            if not raw_mem:
                # If not in SQLite metadata (e.g. placeholder nodes), mock it from graph info
                graph_node = self.graph_store.get_node(node_id)
                if graph_node:
                    raw_mem = {
                        "id": node_id,
                        "memory_type": graph_node.get("type", "entity"),
                        "content": graph_node.get("content", graph_node.get("name", node_id)),
                        "metadata": graph_node,
                        "created_at": now_str,
                        "updated_at": now_str
                    }
                else:
                    continue

            # Update access statistics in metadata
            meta = raw_mem["metadata"]
            access_count = meta.get("access_count", 0) + 1
            meta["access_count"] = access_count
            meta["last_accessed_at"] = now_str
            self.db.update_raw_memory(node_id, raw_mem["content"], meta)

            # Compute similarity score (fallback to 0.1 if not in direct vector search results)
            vector_score = 0.1
            for v_id, score in vector_results:
                if v_id == node_id:
                    vector_score = score
                    break

            # Calculate decayed importance score
            base_importance = meta.get("importance_override", 0.5)
            created_at = raw_mem.get("created_at", now_str)
            last_accessed = meta.get("last_accessed_at", now_str)
            
            decayed_importance = self.scoring_engine.calculate_decayed_importance(
                base_importance=base_importance,
                access_count=access_count,
                created_at_str=created_at,
                last_accessed_at_str=last_accessed,
                current_time_str=now_str
            )

            # Calculate retrieval priority rank
            rank_score = self.scoring_engine.retrieve_priority_score(decayed_importance, vector_score)

            ranked_memories.append({
                "id": node_id,
                "type": raw_mem["memory_type"],
                "content": raw_mem["content"],
                "metadata": meta,
                "rank_score": rank_score
            })

        # Sort all retrieved candidates by priority rank score descending
        ranked_memories.sort(key=lambda x: x["rank_score"], reverse=True)
        top_memories = ranked_memories[:limit]

        # Log search query retrieval for analytics
        retrieved_ids = [m["id"] for m in top_memories]
        self.db.log_retrieval(query, retrieved_ids)

        # Step 4: Compile formatted prompt context
        formatted_parts = []
        for mem in top_memories:
            m_type = mem["type"].capitalize()
            # Beautify based on memory types
            if mem["type"] == "semantic":
                entity = mem["metadata"].get("entity_name")
                topic = mem["metadata"].get("topic")
                label = f"Semantic Fact (Entity: {entity}, Topic: {topic})" if entity else f"Semantic Fact (Topic: {topic})"
            elif mem["type"] == "episodic":
                label = f"Episodic Event (Session: {mem['metadata'].get('session_id')})"
            elif mem["type"] == "procedural":
                label = f"Procedural Workflow (Task: {mem['metadata'].get('task_name')}, Outcome: {mem['metadata'].get('outcome')})"
            else:
                label = f"{m_type} Node"

            formatted_parts.append(f"[{label}] {mem['content']}")

        formatted_context = "\n".join(formatted_parts) if formatted_parts else "No relevant memories found."

        return {
            "memories": top_memories,
            "formatted_context": formatted_context
        }
