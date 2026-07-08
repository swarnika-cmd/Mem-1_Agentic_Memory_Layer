from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from mem1.config import config
from mem1.database import SQLiteDB, SimpleVectorStore, NetworkXGraphStore
from mem1.memory import EpisodicMemoryManager, SemanticMemoryManager, ProceduralMemoryManager
from mem1.engines import ScoringEngine, RetrievalEngine, ConsolidationEngine, EvolutionEngine

app = FastAPI(
    title="Mem1 Cognitive Memory Server",
    description="API for storing, retrieving, and evolving memory layers for AI agents.",
    version="1.0.0"
)

# Initialize storage drivers globally
db = SQLiteDB(db_path=config.DB_PATH)
vector_store = SimpleVectorStore(index_path=config.VECTOR_PATH)
graph_store = NetworkXGraphStore(graph_path=config.GRAPH_PATH)

# Initialize cognitive managers
episodic_mgr = EpisodicMemoryManager(db, vector_store, graph_store)
semantic_mgr = SemanticMemoryManager(db, vector_store, graph_store)
procedural_mgr = ProceduralMemoryManager(db, vector_store, graph_store)

# Initialize cognitive engines
scoring_eng = ScoringEngine(base_decay_rate=config.BASE_DECAY_RATE, relevance_weight=config.RELEVANCE_WEIGHT)
retrieval_eng = RetrievalEngine(db, vector_store, graph_store, scoring_eng)
consolidation_eng = ConsolidationEngine(db, vector_store, graph_store, semantic_mgr)
evolution_eng = EvolutionEngine(db, vector_store, graph_store)

# Pydantic Schemas
class StoreMemoryRequest(BaseModel):
    memory_type: str = Field(..., description="episodic, semantic, or procedural")
    content: str = Field(..., description="Raw text content of the memory")
    session_id: Optional[str] = Field(None, description="Required for episodic memory")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata key-value dictionary")
    
    # Semantic-specific fields
    entity_name: Optional[str] = Field(None, description="Entity linked to semantic fact")
    topic: Optional[str] = Field(None, description="Topic of semantic fact")
    
    # Procedural-specific fields
    task_name: Optional[str] = Field(None, description="Task category name")
    steps: Optional[List[str]] = Field(None, description="Sequential steps of procedure")
    outcome: Optional[str] = Field("success", description="success or failure")

class RetrieveContextRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    limit: Optional[int] = 5

class UpdateMemoryRequest(BaseModel):
    node_id: str
    new_content: str
    additional_metadata: Optional[Dict[str, Any]] = None

# Routes
@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Mem1 Cognitive Memory Layer",
        "version": "1.0.0"
    }

@app.post("/store")
def store_memory(req: StoreMemoryRequest):
    try:
        m_type = req.memory_type.lower()
        if m_type == "episodic":
            if not req.session_id:
                raise HTTPException(status_code=400, detail="session_id is required for episodic memory.")
            node = episodic_mgr.create_episodic_memory(req.content, req.session_id, req.metadata)
        elif m_type == "semantic":
            node = semantic_mgr.create_semantic_memory(req.content, req.entity_name, req.topic, req.metadata)
        elif m_type == "procedural":
            if not req.task_name or not req.steps:
                raise HTTPException(status_code=400, detail="task_name and steps are required for procedural memory.")
            node = procedural_mgr.create_procedural_memory(req.content, req.task_name, req.steps, req.outcome, req.metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid memory_type: '{req.memory_type}'. Must be episodic, semantic, or procedural.")
        
        return {"status": "stored", "memory_id": node.id, "node": node.to_dict()}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
def retrieve_context(req: RetrieveContextRequest):
    try:
        context = retrieval_eng.retrieve_context(
            query=req.query,
            session_id=req.session_id,
            limit=req.limit
        )
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
def update_memory(req: UpdateMemoryRequest):
    try:
        result = evolution_eng.update_memory(
            node_id=req.node_id,
            new_content=req.new_content,
            additional_metadata=req.additional_metadata
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consolidate")
def consolidate_memories(background_tasks: BackgroundTasks):
    try:
        # Run consolidation synchronous for the MVP so it returns results instantly to client
        results = consolidation_eng.consolidate_episodic_memories()
        return {"status": "completed", "consolidated_facts": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Debug & Visualization API endpoints
@app.get("/debug/memories")
def get_all_memories():
    return db.list_all_memories()

@app.get("/debug/graph")
def get_graph_data():
    import networkx as nx
    return nx.readwrite.json_graph.node_link_data(graph_store.graph)

if __name__ == "__main__":
    uvicorn.run("mem1.main:app", host=config.HOST, port=config.PORT, reload=True)
