"""Graph RAG demo using MockEmbedding (no API key required).

Builds a knowledge graph from in-memory entities and relations,
runs vector search and graph traversal, then prints results.

Usage:
    uv run python examples/mock_embedding_demo.py
"""

from __future__ import annotations

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4

# 1. Create an in-memory graph store with small embeddings
store = GrafeoPropertyGraphStore(
    embedding_dimensions=DIMS,
    embedding_metric="cosine",
)

# 2. Insert entities with hand-crafted embeddings
store.upsert_nodes(
    [
        EntityNode(
            name="Python",
            label="language",
            properties={"paradigm": "multi"},
            embedding=[0.9, 0.1, 0.3, 0.2],
        ),
        EntityNode(
            name="Rust",
            label="language",
            properties={"paradigm": "systems"},
            embedding=[0.8, 0.2, 0.4, 0.1],
        ),
        EntityNode(
            name="LlamaIndex",
            label="framework",
            properties={"domain": "LLM"},
            embedding=[0.2, 0.9, 0.1, 0.8],
        ),
        EntityNode(
            name="Grafeo",
            label="database",
            properties={"type": "graph"},
            embedding=[0.3, 0.8, 0.2, 0.7],
        ),
    ]
)

# 3. Insert relations
store.upsert_relations(
    [
        Relation(label="WRITTEN_IN", source_id="Grafeo", target_id="Rust", properties={}),
        Relation(label="BINDINGS_FOR", source_id="Grafeo", target_id="Python", properties={}),
        Relation(label="INTEGRATES", source_id="LlamaIndex", target_id="Grafeo", properties={}),
        Relation(label="WRITTEN_IN", source_id="LlamaIndex", target_id="Python", properties={}),
    ]
)

print(f"Graph: {store.node_count} nodes, {store.edge_count} edges\n")

# 4. Vector search: find nodes closest to a "framework" embedding
print("=== Vector Search ===")
query = VectorStoreQuery(query_embedding=[0.25, 0.85, 0.15, 0.75], similarity_top_k=2)
nodes, scores = store.vector_query(query)
for node, score in zip(nodes, scores, strict=True):
    print(f"  {node.id}: similarity={score:.3f}")

# 5. Graph traversal: 2-hop neighbors from LlamaIndex
print("\n=== 2-hop Neighbors of LlamaIndex ===")
triplets = store.neighbors("LlamaIndex", depth=2)
for src, rel, tgt in triplets:
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

# 6. Schema
print(f"\n=== Schema ===\n{store.get_schema_str()}")

# 7. Summary
print(f"\n=== Summary ===\n{store.summary()}")
