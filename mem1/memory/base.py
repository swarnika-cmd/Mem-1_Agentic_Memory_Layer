from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class MemoryNode(BaseModel):
    """Unified schema representing a cognitive memory node."""
    id: str
    type: str  # episodic, semantic, procedural
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary representation."""
        return self.model_dump()
