"""Hybrid retrieval: structured queries + vector search + graph algorithms.

Shows how to combine LlamaIndex's Property Graph Index with direct access
to Grafeo's query engine and built-in graph algorithms.

Usage:
    pip install grafeo-llamaindex llama-index
    python hybrid_retrieval.py
"""

from __future__ import annotations

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

# ── Build a small in-memory knowledge graph ─────────────────────────────

graph_store = GrafeoPropertyGraphStore(
    embedding_dimensions=4,  # tiny embeddings for demo
    embedding_metric="cosine",
)

graph_store.upsert_nodes(
    [
        EntityNode(
            name="Python",
            label="entity",
            properties={"kind": "language"},
            embedding=[0.9, 0.1, 0.3, 0.2],
        ),
        EntityNode(
            name="Rust",
            label="entity",
            properties={"kind": "language"},
            embedding=[0.8, 0.2, 0.4, 0.1],
        ),
        EntityNode(
            name="LlamaIndex",
            label="entity",
            properties={"kind": "framework"},
            embedding=[0.2, 0.9, 0.1, 0.8],
        ),
        EntityNode(
            name="Grafeo",
            label="entity",
            properties={"kind": "database"},
            embedding=[0.3, 0.8, 0.2, 0.7],
        ),
    ]
)

graph_store.upsert_relations(
    [
        Relation(label="WRITTEN_IN", source_id="Grafeo", target_id="Rust", properties={}),
        Relation(label="BINDINGS_FOR", source_id="Grafeo", target_id="Python", properties={}),
        Relation(label="INTEGRATES", source_id="LlamaIndex", target_id="Grafeo", properties={}),
        Relation(label="WRITTEN_IN", source_id="LlamaIndex", target_id="Python", properties={}),
    ]
)

# ── 1. Structured query (GQL) ───────────────────────────────────────────

print("=== Structured Query ===")
results = graph_store.structured_query("MATCH (a:entity)-[r]->(b:entity) RETURN a.name, r, b.name LIMIT 10")
for row in results:
    print(f"  {row}")

# ── 2. Triplet retrieval ────────────────────────────────────────────────

print("\n=== Triplets for Grafeo ===")
triplets = graph_store.get_triplets(entity_names=["Grafeo"])
for src, rel, tgt in triplets:
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

# ── 3. Multi-hop relationship map ───────────────────────────────────────

print("\n=== 2-hop Relationship Map from LlamaIndex ===")
seeds = graph_store.get(ids=["LlamaIndex"])
rel_map = graph_store.get_rel_map(seeds, depth=2)
for src, rel, tgt in rel_map:
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

# ── 4. Vector similarity search ─────────────────────────────────────────

print("\n=== Vector Search (closest to AI/framework space) ===")
query = VectorStoreQuery(query_embedding=[0.25, 0.85, 0.15, 0.75], similarity_top_k=2)
nodes, scores = graph_store.vector_query(query)
for node, score in zip(nodes, scores, strict=True):
    print(f"  {node.id}: similarity={score:.3f}")

# ── 5. Direct access to Grafeo's graph algorithms ───────────────────────

print("\n=== PageRank (via Grafeo) ===")
db = graph_store.client
pr_scores = db.algorithms.pagerank()
# pagerank() returns a dict mapping node_id → score
for node_id, score in sorted(pr_scores.items(), key=lambda x: x[1], reverse=True):
    node = db.get_node(node_id)
    name = node.properties().get("name", f"node-{node_id}")
    print(f"  {name}: {score:.4f}")

# ── 6. Schema inspection ────────────────────────────────────────────────

print("\n=== Schema ===")
print(graph_store.get_schema_str())
