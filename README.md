# 🧠 Mem1: Cognitive Memory Layer for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Cognitive--Graph-orange)](#system-architecture)

> **"The goal is not to store memory. The goal is to build a system where memory evolves into structured knowledge that improves intelligence over time."**

Mem1 is a structured, persistent, and evolving memory system designed to go beyond traditional vector-based memory. Instead of treating memory as passive vector storage, Mem1 models memory as an **active, structured, and adaptive system** inspired by human cognitive science. It enables AI agents to store, retrieve, reason over, and consolidate memories across sessions.

---

## 🚀 Key Innovations

### 1. Structured Memory Graph (Core Novelty)
Rather than a flat vector database, Mem1 builds a **Semantic-Episodic Knowledge Graph**.
* **Nodes**: *Events* (temporal experiences), *Concepts* (abstract ideas, rules), and *Entities* (people, places, tools, sessions).
* **Edges**: Directed relationships such as `related_to`, `derived_from`, `caused_by`, and `contradicts`.

```
                  [Entity: User] 
                        │
                        │ (interacted_in)
                        ▼
                 [Event: Chat Session 1] ──(caused_by)──► [Event: Task Failure]
                        │                                      │
                        │ (discussed)                          │ (derived_from)
                        ▼                                      ▼
               [Concept: SQLite Locking] ◄─────────────── [Concept: DB Timeout]
```

### 2. Multi-Type Memory System
Mem1 splits memory into three cognitive storage structures:
* **Episodic Memory**: Raw logs, exact dialogues, and timeline events of agent interactions.
* **Semantic Memory**: Abstracted facts, learned concepts, and entity properties extracted from experiences.
* **Procedural Memory**: Worked plans, tool execution workflows, and task success/failure trajectories.

### 3. Memory Consolidation Engine
Mem1 runs background workers to periodically:
* Merge overlapping or highly similar memories.
* Extract generalized rules (Episodic $\rightarrow$ Semantic transition).
* Prune redundant details to limit context window bloat.

### 4. Dynamic Importance Scoring
Every memory node has an evolving importance score calculated using:
$$\text{Importance} = w_r \cdot \text{Recency} + w_f \cdot \text{Frequency} + w_v \cdot \text{Relevance} + w_u \cdot \text{Usage}$$
* **Forgetting Mechanism**: Low-importance nodes decay over time and are pruned, while high-frequency nodes are reinforced.

### 5. Contextual Hybrid Retrieval
A multi-stage retrieval pipeline combines:
1. **Vector Similarity**: Fast semantic search of node embeddings.
2. **Graph Traversal**: Relational multi-hop expansions (e.g., retrieving related concepts or previous failure states).
3. **Intent Understanding**: Disambiguating queries based on user context.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                               │
│        [store_memory()]    [retrieve_context()]    [update_memory()]   │
└───────────────────┬───────────────────▲───────────────────┬────────────┘
                    │                   │                   │
                    ▼                   │                   ▼
┌───────────────────────────────────────┴────────────────────────────────┐
│                           API LAYER (FastAPI)                          │
└───────────────────┬───────────────────────────────────────▲────────────┘
                    │                                       │
                    ▼                                       │
┌───────────────────────────────────────────────────────────┴────────────┐
│                       COGNITIVE RETRIEVAL ENGINE                       │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌──────────────┐  │
│  │   Vector Similarity   │ │    Graph Traversal    │ │ Intent Parser│  │
│  └───────────┬───────────┘ └───────────┬───────────┘ └──────┬───────┘  │
└──────────────┼─────────────────────────┼────────────────────┼──────────┘
               ▼                         ▼                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                            DATABASE STORAGE                            │
│  ┌───────────────────────┐ ┌───────────────────────┐ ┌──────────────┐  │
│  │     Vector Store      │ │   SQLite / Postgres   │ │ NetworkX/Neo4│  │
│  │    (Chroma / NumPy)   │ │  (Relational Metadata)│ │  (Graph DB)  │  │
│  └───────────────────────┘ └───────────────────────┘ └──────────────┘  │
└───────────────────────────────────────▲────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴────────────────────────────────┐
│                      BACKGROUND CONSOLIDATION ENGINE                   │
│   ┌────────────────────┐   ┌────────────────────┐   ┌──────────────┐   │
│   │ Importance Decay   │   │ Entity Resolution  │   │ Fact Merger  │   │
│   └────────────────────┘   └────────────────────┘   └──────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Layout

```text
mem1/
├── database/            # Storage abstractions and local/remote drivers
│   ├── base.py          # Abstract interfaces for database adaptors
│   ├── sqlite_db.py     # SQLite driver for relational metadata and logs
│   ├── vector_store.py  # Local vector storage with cosine similarity
│   └── graph_store.py   # NetworkX graph store mapping nodes & relations
├── memory/              # Memory structures
│   ├── base.py          # Unified memory container
│   ├── episodic.py      # Episodic memory manager
│   ├── semantic.py      # Semantic memory manager
│   └── procedural.py    # Procedural memory manager
├── engines/             # Cognitive processing pipelines
│   ├── scoring.py       # Relevance and recency decay calculator
│   ├── retrieval.py     # Multi-stage graph-vector retrieval engine
│   ├── consolidation.py # Background facts extractor & summarizer
│   └── evolution.py     # Memory collision and updates manager
├── sdk/                 # Client library for agent integration
│   └── client.py        # Mem1Client wrapper
├── dashboard/           # Memory Debugger & Visualizer
│   └── app.py           # Streamlit dashboard script
├── main.py              # FastAPI server entry point
└── config.py            # Global environment and configuration manager
```

---

## ⚡ Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/swarnika-cmd/Mem-1_Agentic_Memory_Layer.git
cd Mem-1_Agentic_Memory_Layer
pip install -r requirements.txt
```

### 2. Launch FastAPI Service
Start the cognitive server locally:
```bash
uvicorn mem1.main:app --host 127.0.0.1 --port 8000
```

### 3. Launch Memory Debugger Dashboard
Open the interactive Streamlit visualizer to inspect your agent's memory graph in real-time:
```bash
streamlit run mem1/dashboard/app.py
```

---

## 💡 Developer SDK Usage

You can connect your agent to Mem1 using our lightweight Python client in just a few lines of code:

```python
from mem1.sdk.client import Mem1Client

# Initialize client
client = Mem1Client(base_url="http://localhost:8000")

# 1. Store an episodic conversation memory
client.store_memory(
    memory_type="episodic",
    content="User explained their project requires SQLite because it needs to be zero-dependency.",
    metadata={"session_id": "session_99", "importance_override": 0.8}
)

# 2. Store a semantic fact (abstracted from episodic events)
client.store_memory(
    memory_type="semantic",
    content="Project requires SQLite for zero-dependency portability.",
    metadata={"entity": "Project", "topic": "Database"}
)

# 3. Retrieve context for a new query
context = client.retrieve_context(
    query="What database should I use for their project?",
    session_id="session_99",
    limit=5
)

print(context["formatted_context"])
# Output:
# [Semantic Fact] Project requires SQLite for zero-dependency portability.
# [Episodic Event] User explained their project requires SQLite because it needs to be zero-dependency.
```

---

## 🛡️ Access to Frontier Reasoning Models

### Why Mem1 Requires Advanced Reasoning Capabilities (e.g. Gemini 1.5 Pro / GPT-4o / Claude 3.5 Sonnet)
To realize its core vision, Mem1 requires advanced frontier reasoning capabilities for several background cognitive tasks:
1. **Fact Abstraction (Episodic $\rightarrow$ Semantic)**: Transforming raw chat transcripts (episodic memory) into structured facts (semantic memory) without losing context or introducing hallucinations.
2. **Entity & Relation Extraction**: Dynamically reading incoming text and generating graph nodes (Entities, Concepts, Events) and edges (`caused_by`, `contradicts`, `related_to`) to update the knowledge graph accurately.
3. **Conflict Resolution**: Real-time resolution of contradictory memories (e.g., if a user changes their requirements mid-session) and managing version history.
4. **Graph-based Intent Parsing**: Resolving multi-hop agent queries into sub-queries that execute across both vector matches and path traversals.

For inquiries regarding research API access, benchmark integrations, or developer credits, contact the team at [Mem1 GitHub Issues](https://github.com/swarnika-cmd/Mem-1_Agentic_Memory_Layer/issues).
