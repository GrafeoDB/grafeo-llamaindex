"""Basic Graph RAG with Grafeo and LlamaIndex.

Builds a Property Graph Index from documents, then queries it
with graph-aware retrieval. All data lives in a single .db file.

Usage:
    pip install grafeo-llamaindex llama-index
    export OPENAI_API_KEY=sk-...
    python basic_graph_rag.py
"""

from __future__ import annotations

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader

from grafeo_llamaindex import GrafeoPropertyGraphStore

# 1. Load documents (replace with your own data directory)
documents = SimpleDirectoryReader("./data").load_data()

# 2. Create Grafeo-backed graph store
graph_store = GrafeoPropertyGraphStore(
    db_path="./knowledge_graph.db",
    embedding_dimensions=1536,
)

# 3. Build the Property Graph Index — extracts entities, relations, and embeddings
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,
    show_progress=True,
)

# 4. Query with graph-aware retrieval
retriever = index.as_retriever(include_text=True)
nodes = retriever.retrieve("What are the key relationships in this dataset?")
for node in nodes:
    print(node.text)

# 5. The graph persists automatically — reopen later with:
#    graph_store = GrafeoPropertyGraphStore(db_path="./knowledge_graph.db")
#    index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)
