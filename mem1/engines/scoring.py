import math
from datetime import datetime
from typing import Dict, Any

class ScoringEngine:
    """Calculates cognitive importance and decay for memories over time."""

    def __init__(self, base_decay_rate: float = 0.05, relevance_weight: float = 0.4):
        self.base_decay_rate = base_decay_rate
        self.relevance_weight = relevance_weight

    def calculate_decayed_importance(
        self, 
        base_importance: float, 
        access_count: int, 
        created_at_str: str, 
        last_accessed_at_str: str,
        current_time_str: Optional[str] = None
    ) -> float:
        """Calculates decayed importance of a memory node based on time and usage.
        
        Uses the formula: Decayed Importance = base_importance * exp(-lambda * delta_t)
        Where lambda (decay rate) decreases as access_count increases (spacing effect).
        """
        now = datetime.fromisoformat(current_time_str) if current_time_str else datetime.utcnow()
        last_accessed = datetime.fromisoformat(last_accessed_at_str)
        
        # Calculate time delta in hours
        delta_hours = max(0.0, (now - last_accessed).total_seconds() / 3600.0)
        
        # Spacing effect: access frequency reduces the rate of forgetting (decay_rate)
        # e.g., if access_count = 0 -> lambda = base_decay_rate
        # if access_count = 10 -> lambda = base_decay_rate / 3
        effective_decay_rate = self.base_decay_rate / (1.0 + math.log1p(access_count))
        
        # Exponential decay calculation
        decay_factor = math.exp(-effective_decay_rate * delta_hours)
        decayed_score = base_importance * decay_factor
        
        # Usage boost: reward nodes that have been accessed repeatedly
        frequency_boost = 0.1 * min(5.0, access_count)
        
        final_score = min(1.0, max(0.0, decayed_score + frequency_boost))
        return final_score

    def retrieve_priority_score(
        self, 
        node_importance: float, 
        vector_similarity: float
    ) -> float:
        """Combine static importance score with query relevance similarity score."""
        return (1.0 - self.relevance_weight) * node_importance + self.relevance_weight * vector_similarity
