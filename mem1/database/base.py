from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class BaseRelationalDB(ABC):
    """Abstract interface for relational metadata and session logs storage (e.g. SQLite, PostgreSQL)."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the relational database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection to the database."""
        pass

    @abstractmethod
    def store_raw_memory(self, memory_id: str, memory_type: str, content: str, metadata: Dict[str, Any]) -> None:
        """Store raw memory entry with metadata."""
        pass

    @abstractmethod
    def get_raw_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve raw memory entry by its ID."""
        pass

    @abstractmethod
    def update_raw_memory(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Update an existing raw memory entry."""
        pass

    @abstractmethod
    def delete_raw_memory(self, memory_id: str) -> None:
        """Delete a raw memory entry by ID."""
        pass

    @abstractmethod
    def log_retrieval(self, query: str, retrieved_ids: List[str], feedback: Optional[float] = None) -> None:
        """Log retrieval operations for debugging and reinforcement learning."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for vector similarity index storage (e.g. Local NumPy, FAISS, Chroma)."""

    @abstractmethod
    def add_vector(self, node_id: str, text: str, embedding: Optional[List[float]] = None) -> None:
        """Add a text entry and its vector representation to the index."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Search similar items returning a list of (node_id, similarity_score)."""
        pass

    @abstractmethod
    def remove_vector(self, node_id: str) -> None:
        """Remove a vector representation by node ID."""
        pass


class BaseGraphStore(ABC):
    """Abstract interface for graph storage (e.g. NetworkX, Neo4j, ArangoDB)."""

    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
        """Add a node representing an Entity, Concept, or Event."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node details and properties."""
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update node properties."""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and its associated edges."""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> None:
        """Create a directed relation between two memory nodes."""
        pass

    @abstractmethod
    def get_edges(self, node_id: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming/outgoing edges for a node."""
        pass

    @abstractmethod
    def delete_edge(self, source_id: str, target_id: str, relation_type: str) -> None:
        """Delete a specific relation between two nodes."""
        pass

    @abstractmethod
    def traverse_graph(self, start_node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Traverse the graph from a start node to fetch context within a certain depth."""
        pass
