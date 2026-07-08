import os

class Config:
    """Global configuration settings for the Mem1 memory layer."""
    
    DB_PATH = os.environ.get("MEM1_DB_PATH", "mem1_local.db")
    VECTOR_PATH = os.environ.get("MEM1_VECTOR_PATH", "mem1_vector_index.json")
    GRAPH_PATH = os.environ.get("MEM1_GRAPH_PATH", "mem1_graph.json")
    
    # API configuration
    HOST = os.environ.get("MEM1_HOST", "127.0.0.1")
    PORT = int(os.environ.get("MEM1_PORT", "8000"))
    
    # Cognitive thresholds
    BASE_DECAY_RATE = float(os.environ.get("MEM1_DECAY_RATE", "0.05"))
    RELEVANCE_WEIGHT = float(os.environ.get("MEM1_RELEVANCE_WEIGHT", "0.4"))

config = Config()
