import json
import os
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from mem1.database.base import BaseGraphStore

class NetworkXGraphStore(BaseGraphStore):
    """NetworkX implementation for local Graph Storage.
    
    Persists graph to a JSON file using node-link format.
    """

    def __init__(self, graph_path: str = "mem1_graph.json"):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self.load_graph()

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
        """Add a node representing an Entity, Concept, or Event."""
        props = properties.copy()
        props["type"] = node_type
        self.graph.add_node(node_id, **props)
        self.save_graph()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node details and properties."""
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.nodes[node_id])

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Update node properties."""
        if self.graph.has_node(node_id):
            # Keep the original node type if not overwritten
            node_type = self.graph.nodes[node_id].get("type", "concept")
            self.graph.nodes[node_id].update(properties)
            self.graph.nodes[node_id]["type"] = properties.get("type", node_type)
            self.save_graph()

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its associated edges."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            self.save_graph()

    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> None:
        """Create a directed relation between two memory nodes."""
        # Ensure source and target nodes exist in the graph (scaffold them if they don't)
        if not self.graph.has_node(source_id):
            self.graph.add_node(source_id, type="entity")
        if not self.graph.has_node(target_id):
            self.graph.add_node(target_id, type="concept")
            
        self.graph.add_edge(source_id, target_id, relation=relation_type, weight=weight)
        self.save_graph()

    def get_edges(self, node_id: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming/outgoing edges for a node."""
        edges = []
        if not self.graph.has_node(node_id):
            return edges

        # Outgoing edges
        for target in self.graph.successors(node_id):
            data = self.graph.get_edge_data(node_id, target)
            edges.append((node_id, target, data))
        
        # Incoming edges
        for source in self.graph.predecessors(node_id):
            data = self.graph.get_edge_data(source, node_id)
            edges.append((source, node_id, data))

        return edges

    def delete_edge(self, source_id: str, target_id: str, relation_type: str) -> None:
        """Delete a specific relation between two nodes."""
        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)
            self.save_graph()

    def traverse_graph(self, start_node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Traverse the graph using Breadth-First Search up to depth N to compile context."""
        if not self.graph.has_node(start_node_id):
            return []

        visited = set()
        queue = [(start_node_id, 0)]
        results = []

        while queue:
            node_id, current_depth = queue.pop(0)
            if node_id in visited:
                continue

            visited.add(node_id)
            node_data = self.get_node(node_id)
            if node_data:
                results.append({
                    "id": node_id,
                    "depth": current_depth,
                    **node_data
                })

            if current_depth < depth:
                # Traverse outgoing edges
                for neighbor in self.graph.successors(node_id):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
                
                # Traverse incoming edges as well for a richer contextual neighborhood
                for neighbor in self.graph.predecessors(node_id):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))

        return results

    def save_graph(self) -> None:
        """Serialize and save NetworkX graph to a JSON file."""
        # Ensure parent directories exist
        graph_dir = os.path.dirname(self.graph_path)
        if graph_dir and not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
            
        data = nx.readwrite.json_graph.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_graph(self) -> None:
        """Deserialize and load NetworkX graph from a JSON file."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.readwrite.json_graph.node_link_graph(data)
            except Exception:
                # Re-initialize empty graph if file read fails
                self.graph = nx.DiGraph()
