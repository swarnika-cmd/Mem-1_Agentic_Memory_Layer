import os
import pytest
from datetime import datetime

from mem1.database import SQLiteDB, SimpleVectorStore, NetworkXGraphStore
from mem1.memory import EpisodicMemoryManager, SemanticMemoryManager, ProceduralMemoryManager
from mem1.engines import ScoringEngine, RetrievalEngine, ConsolidationEngine, EvolutionEngine

TEST_DB = "test_mem1_local.db"
TEST_VECTOR = "test_mem1_vector_index.json"
TEST_GRAPH = "test_mem1_graph.json"

@pytest.fixture(autouse=True)
def cleanup_files():
    """Fixture to ensure a fresh set of databases for each test run."""
    # Clean up before
    for f in [TEST_DB, TEST_VECTOR, TEST_GRAPH]:
        if os.path.exists(f):
            os.remove(f)
            
    yield
    
    # Clean up after
    for f in [TEST_DB, TEST_VECTOR, TEST_GRAPH]:
        if os.path.exists(f):
            os.remove(f)

def test_full_memory_lifecycle():
    # 1. Initialize databases
    db = SQLiteDB(db_path=TEST_DB)
    vector = SimpleVectorStore(index_path=TEST_VECTOR)
    graph = NetworkXGraphStore(graph_path=TEST_GRAPH)

    # 2. Initialize Managers
    episodic_mgr = EpisodicMemoryManager(db, vector, graph)
    semantic_mgr = SemanticMemoryManager(db, vector, graph)
    procedural_mgr = ProceduralMemoryManager(db, vector, graph)

    # 3. Store Episodic Memories
    node_epi1 = episodic_mgr.create_episodic_memory(
        content="I prefer using SQLite because it is self-contained.",
        session_id="sess_abc"
    )
    node_epi2 = episodic_mgr.create_episodic_memory(
        content="Our SQLite database requires zero-dependency deployment portability.",
        session_id="sess_abc"
    )

    assert node_epi1.id.startswith("epi_")
    assert node_epi2.id.startswith("epi_")

    # Verify SQLite persistence
    mem1_row = db.get_raw_memory(node_epi1.id)
    assert mem1_row is not None
    assert mem1_row["content"] == "I prefer using SQLite because it is self-contained."

    # Verify Graph nodes & edges
    assert graph.graph.has_node(node_epi1.id)
    assert graph.graph.has_node(node_epi2.id)
    # Check if temporal followed_by edge was created between events
    assert graph.graph.has_edge(node_epi1.id, node_epi2.id)

    # 4. Store Semantic facts
    node_sem = semantic_mgr.create_semantic_memory(
        content="SQLite is a zero-dependency relational database library.",
        entity_name="SQLite",
        topic="database_architecture"
    )
    assert node_sem.id.startswith("sem_")
    assert graph.graph.has_node("ent_sqlite")
    assert graph.graph.has_edge("ent_sqlite", node_sem.id)

    # 5. Store Procedural Memory
    node_pro = procedural_mgr.create_procedural_memory(
        content="Configure database connection in settings",
        task_name="DB Initialization",
        steps=["Create file", "Define schema", "Open sqlite3 connection"],
        outcome="success"
    )
    assert node_pro.id.startswith("pro_")

    # 6. Test Retrieval Engine
    retrieval_eng = RetrievalEngine(db, vector, graph)
    retrieved = retrieval_eng.retrieve_context("need sqlite setup", limit=3)
    
    assert len(retrieved["memories"]) > 0
    # The formatted context should contain SQLite reference
    assert "sqlite" in retrieved["formatted_context"].lower()

    # 7. Test Consolidation Engine (Episodic -> Semantic Fact abstraction)
    consolidation_eng = ConsolidationEngine(db, vector, graph, semantic_mgr)
    new_facts = consolidation_eng.consolidate_episodic_memories()
    
    assert len(new_facts) > 0
    # Verify that the original episodic memory is now marked as consolidated
    updated_epi = db.get_raw_memory(node_epi1.id)
    assert updated_epi["metadata"].get("consolidated") is True

    # 8. Test Evolution (Updates & Versioning & Contradictions)
    evolution_eng = EvolutionEngine(db, vector, graph)
    # Perform update
    update_res = evolution_eng.update_memory(
        node_id=node_sem.id,
        new_content="SQLite is no longer zero-dependency, it requires locking extensions.",
        additional_metadata={"update_reason": "Change in specs"}
    )
    
    assert update_res["version"] == 2
    # Verify history tracking
    final_node = db.get_raw_memory(node_sem.id)
    assert len(final_node["metadata"]["history"]) == 1
    assert final_node["metadata"]["history"][0]["content"] == "SQLite is a zero-dependency relational database library."

    # Verify contradiction edge in graph (reversals check)
    # SQLite no longer zero-dependency vs SQLite is a zero-dependency
    assert graph.graph.has_edge(node_sem.id, node_sem.id)
    assert graph.graph.get_edge_data(node_sem.id, node_sem.id)["relation"] == "contradicts"
    
    db.close()
