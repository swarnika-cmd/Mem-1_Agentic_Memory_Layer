import requests
from typing import Dict, Any, List, Optional

class Mem1Client:
    """Python Client SDK for interacting with the Mem1 Cognitive Memory Server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def store_memory(
        self,
        memory_type: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
        topic: Optional[str] = None,
        task_name: Optional[str] = None,
        steps: Optional[List[str]] = None,
        outcome: Optional[str] = "success"
    ) -> Dict[str, Any]:
        """Store an episodic, semantic, or procedural memory entry."""
        url = f"{self.base_url}/store"
        payload = {
            "memory_type": memory_type,
            "content": content,
            "session_id": session_id,
            "metadata": metadata,
            "entity_name": entity_name,
            "topic": topic,
            "task_name": task_name,
            "steps": steps,
            "outcome": outcome
        }
        # Remove None values to clean up payload
        payload = {k: v for k, v in payload.items() if v is not None}
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def retrieve_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Query the hybrid retrieval engine to fetch contextual prompt memory injection."""
        url = f"{self.base_url}/retrieve"
        payload = {
            "query": query,
            "session_id": session_id,
            "limit": limit
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def update_memory(
        self,
        node_id: str,
        new_content: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing memory entry and keep track of its history."""
        url = f"{self.base_url}/update"
        payload = {
            "node_id": node_id,
            "new_content": new_content,
            "additional_metadata": additional_metadata
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def consolidate(self) -> Dict[str, Any]:
        """Trigger background consolidation routine (merging episodic dialogues to semantic rules)."""
        url = f"{self.base_url}/consolidate"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
