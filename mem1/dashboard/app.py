import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Setup page layout
st.set_page_config(
    page_title="Mem1 Cognitive Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 16px;
    }
    div[data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.title("🧠 Mem1: Cognitive Memory Debugging & Visualizer")
st.caption("Active monitoring and inspection interface for Agentic Memory Networks")

# Sidebar settings
st.sidebar.title("Configuration")
api_endpoint = st.sidebar.text_input("Mem1 API Server", value=API_URL)

# Test API connection
server_online = False
try:
    health_resp = requests.get(f"{api_endpoint}/")
    if health_resp.status_code == 200:
        server_online = True
        st.sidebar.success("Memory Server: ONLINE")
        system_info = health_resp.json()
        st.sidebar.info(f"System: {system_info.get('system')}\nVersion: {system_info.get('version')}")
except Exception:
    st.sidebar.error("Memory Server: OFFLINE\nPlease start: uvicorn mem1.main:app")

if server_online:
    # Fetch data
    try:
        memories_resp = requests.get(f"{api_endpoint}/debug/memories")
        memories = memories_resp.json() if memories_resp.status_code == 200 else []
        
        graph_resp = requests.get(f"{api_endpoint}/debug/graph")
        graph_data = graph_resp.json() if graph_resp.status_code == 200 else {"nodes": [], "links": []}
    except Exception as e:
        memories = []
        graph_data = {"nodes": [], "links": []}
        st.error(f"Error loading system data: {e}")

    # Stat metrics
    episodic_count = sum(1 for m in memories if m["memory_type"] == "episodic")
    semantic_count = sum(1 for m in memories if m["memory_type"] == "semantic")
    procedural_count = sum(1 for m in memories if m["memory_type"] == "procedural")
    total_relations = len(graph_data.get("links", []))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Episodic Memories (Events)", episodic_count)
    col2.metric("Semantic Memories (Facts)", semantic_count)
    col3.metric("Procedural Memories (Tasks)", procedural_count)
    col4.metric("Active Graph Relations (Edges)", total_relations)

    # Tabs
    tab_viz, tab_search, tab_store, tab_consolidate = st.tabs([
        "🔮 Memory Graph Network",
        "🔍 Contextual Retrieval Debugger",
        "📥 SDK Sandbox (Store Memory)",
        "⚙️ Cognitive Consolidation & Evolution"
    ])

    # 1. GRAPH VISUALIZATION TAB
    with tab_viz:
        st.header("Semantic-Episodic Knowledge Network")
        st.markdown("Dynamic visualization of Entity, Concept, Event, and Procedure nodes linked by cognitive relationships.")
        
        if graph_data["nodes"]:
            # Construct NetworkX graph from server JSON data
            G = nx.DiGraph()
            node_types = {}
            for node in graph_data["nodes"]:
                node_id = node["id"]
                node_type = node.get("type", "concept")
                node_types[node_id] = node_type
                
                # Truncate text content for label clarity
                content = node.get("content", node.get("name", node_id))
                label = content[:20] + "..." if len(content) > 20 else content
                G.add_node(node_id, label=label, type=node_type)
            
            for edge in graph_data["links"]:
                G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "related_to"))

            # Plot layout
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0f111a')
            pos = nx.spring_layout(G, k=0.3, seed=42)
            
            # Draw color-coded node categories
            colors = {"event": "#3b82f6", "concept": "#f97316", "entity": "#10b981", "procedure": "#a855f7"}
            node_colors = [colors.get(node_types[n], "#6b7280") for n in G.nodes()]
            
            labels = nx.get_node_attributes(G, 'label')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#4b5563", width=1.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#f8fafc", font_weight="bold", ax=ax)
            
            # Legend
            for name, col in colors.items():
                ax.scatter([], [], c=col, label=name.capitalize())
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
            
            ax.axis('off')
            st.pyplot(fig)
            
            # Show interactive tabular view
            st.subheader("Memory Nodes Index")
            node_df = pd.DataFrame(graph_data["nodes"])
            if not node_df.empty:
                st.dataframe(node_df, use_container_width=True)
        else:
            st.warning("The memory graph is currently empty. Use the SDK Sandbox to inject memories!")

    # 2. CONTEXTUAL RETRIEVAL DEBUGGER TAB
    with tab_search:
        st.header("Contextual Hybrid Search Trace")
        st.markdown("Inspect how the retrieval engine combines vector similarity and graph neighbors, decay-weights them, and builds agent context.")
        
        search_query = st.text_input("Enter Agent Query (e.g. 'sqlite zero dependency')", value="sqlite zero dependency")
        search_session = st.text_input("Session ID Filter (Optional)", value="")
        search_limit = st.slider("Max Context Nodes to Retrieve", min_value=1, max_value=10, value=3)

        if st.button("Query Retrieval Engine", key="btn_query"):
            with st.spinner("Executing retrieval trace..."):
                payload = {
                    "query": search_query,
                    "session_id": search_session if search_session else None,
                    "limit": search_limit
                }
                res = requests.post(f"{api_endpoint}/retrieve", json=payload)
                if res.status_code == 200:
                    retrieval_res = res.json()
                    
                    st.subheader("💡 Prompt Context Injected to Agent")
                    st.code(retrieval_res["formatted_context"], language="text")
                    
                    st.subheader("📊 Retrieval Scoring & Decayed Importance Trace")
                    memories_retrieved = retrieval_res.get("memories", [])
                    if memories_retrieved:
                        trace_df = pd.DataFrame([
                            {
                                "ID": m["id"],
                                "Type": m["type"],
                                "Content": m["content"],
                                "Rank Priority": round(m["rank_score"], 4),
                                "Base Importance": m["metadata"].get("importance_override", 0.5),
                                "Access Count": m["metadata"].get("access_count", 1),
                                "Version": m["metadata"].get("version", 1)
                            }
                            for m in memories_retrieved
                        ])
                        st.dataframe(trace_df, use_container_width=True)
                    else:
                        st.write("No matching candidate nodes detected.")
                else:
                    st.error("Error connecting to retrieval endpoint.")

    # 3. SDK SANDBOX (STORE MEMORY)
    with tab_store:
        st.header("Agent SDK Simulator")
        st.markdown("Manually inject and test episodic, semantic, or procedural memory packets.")
        
        m_type = st.selectbox("Memory Type", ["Episodic (Event/Conversation)", "Semantic (Fact/Concept)", "Procedural (Plan/Workflow)"])
        content = st.text_area("Memory Content Text", placeholder="Describe the agent event, fact, or workflow outcome...")
        
        # Memory-specific inputs
        if m_type == "Episodic (Event/Conversation)":
            sess_id = st.text_input("Session ID", value="session_01")
            importance = st.slider("Importance Override", min_value=0.1, max_value=1.0, value=0.7)
            
            if st.button("Store Episodic Memory"):
                payload = {
                    "memory_type": "episodic",
                    "content": content,
                    "session_id": sess_id,
                    "metadata": {"importance_override": importance}
                }
                res = requests.post(f"{api_endpoint}/store", json=payload)
                if res.status_code == 200:
                    st.success(f"Stored successfully! ID: {res.json()['memory_id']}")
                    st.rerun()
                else:
                    st.error("Failed to store memory.")
                    
        elif m_type == "Semantic (Fact/Concept)":
            entity = st.text_input("Target Entity (e.g. 'User', 'SQLite')", value="SQLite")
            topic = st.text_input("Topic Category", value="database_portability")
            
            if st.button("Store Semantic Fact"):
                payload = {
                    "memory_type": "semantic",
                    "content": content,
                    "entity_name": entity,
                    "topic": topic
                }
                res = requests.post(f"{api_endpoint}/store", json=payload)
                if res.status_code == 200:
                    st.success(f"Stored successfully! ID: {res.json()['memory_id']}")
                    st.rerun()
                else:
                    st.error("Failed to store memory.")
                    
        elif m_type == "Procedural (Plan/Workflow)":
            task_name = st.text_input("Task Name", value="Setup Project Repo")
            steps_input = st.text_area("Workflow Steps (comma-separated)", value="Create Folder, Git Init, Add Files, Git Commit")
            outcome = st.selectbox("Execution Outcome", ["success", "failure"])
            
            if st.button("Store Procedural Workflow"):
                steps_list = [s.strip() for s in steps_input.split(",") if s.strip()]
                payload = {
                    "memory_type": "procedural",
                    "content": content,
                    "task_name": task_name,
                    "steps": steps_list,
                    "outcome": outcome
                }
                res = requests.post(f"{api_endpoint}/store", json=payload)
                if res.status_code == 200:
                    st.success(f"Stored successfully! ID: {res.json()['memory_id']}")
                    st.rerun()
                else:
                    st.error("Failed to store memory.")

    # 4. COGNITIVE CONSOLIDATION & EVOLUTION
    with tab_consolidate:
        st.header("Consolidation Engine & Evolution Tracker")
        
        col_c, col_e = st.columns(2)
        
        with col_c:
            st.subheader("Memory Consolidation")
            st.markdown("Consolidation abstractively gathers episodic memories, extracts semantic rules/concepts, and connects them with `derived_from` edges.")
            if st.button("Trigger Consolidation Sweeper"):
                res = requests.post(f"{api_endpoint}/consolidate")
                if res.status_code == 200:
                    result = res.json()
                    facts = result.get("consolidated_facts", [])
                    st.success(f"Consolidation pass completed. Consynthesized {len(facts)} new semantic concepts.")
                    if facts:
                        for f in facts:
                            st.info(f"Consolidated Fact: {f['fact']}\n(Sources: {', '.join(f['sources'])})")
                        st.rerun()
                else:
                    st.error("Failed to run consolidation.")

        with col_e:
            st.subheader("Memory Evolution (Update & Versions)")
            st.markdown("Modify an existing memory node. This generates a history log stack and assesses contradiction transitions.")
            
            # Retrieve node options
            node_ids = [m["id"] for m in memories]
            if node_ids:
                selected_node_id = st.selectbox("Select Memory Node to Update", node_ids)
                selected_node = next(m for m in memories if m["id"] == selected_node_id)
                
                st.text("Current Content:")
                st.code(selected_node["content"], language="text")
                
                new_text = st.text_area("New Updated Content", value=selected_node["content"])
                reason = st.text_input("Change Reason / Update Log", value="Requirements modification")
                
                if st.button("Evolve Node Content"):
                    payload = {
                        "node_id": selected_node_id,
                        "new_content": new_text,
                        "additional_metadata": {"update_reason": reason}
                    }
                    res = requests.post(f"{api_endpoint}/update", json=payload)
                    if res.status_code == 200:
                        st.success("Memory evolved successfully! Version incremented.")
                        st.rerun()
                    else:
                        st.error("Failed to update memory node.")
                        
                # Version History
                history = selected_node["metadata"].get("history", [])
                if history:
                    st.subheader("📜 Version Changelog History")
                    st.write(history)
            else:
                st.info("No nodes available to evolve.")
else:
    st.warning("Ensure the FastAPI backend is running before exploring the visualizer.")
