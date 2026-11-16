"""Hybrid retrieval on the Silicon Valley graph: structured queries, vector search and PageRank.

Shows how to combine the LlamaIndex property graph store with direct access to Grafeo's query
engine and built-in graph algorithms. Uses the Silicon Valley (Pied Piper) dataset from
grafeo.ai with the MiniLM vectors it ships, so it needs no API key or model.

Usage:
    uv add grafeo-llamaindex "grafeo-datasets[grafeo]"
    uv run python hybrid_retrieval.py
"""

from __future__ import annotations

from grafeo_datasets import silicon_valley
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

# ── Load the Silicon Valley graph ───────────────────────────────────────

vectors = silicon_valley.vectors()
graph_store = GrafeoPropertyGraphStore(embedding_dimensions=vectors.dimension)

nodes = silicon_valley.nodes()
names = {n["key"]: n["name"] for n in nodes}
graph_store.upsert_nodes(
    [
        EntityNode(
            name=n["name"],
            label=n["label"],
            properties={"description": n["description"], **n["props"]},
            embedding=[float(x) for x in vectors.by_key[n["key"]]],
        )
        for n in nodes
    ]
)
graph_store.upsert_relations(
    [
        Relation(label=e["type"], source_id=names[e["source"]], target_id=names[e["target"]], properties=e["props"])
        for e in silicon_valley.edges()
    ]
)

# ── Structured queries (Cypher by default, GQL on request) ──────────────

print("=== How is Son of Anton related to Anton? ===")
rows = graph_store.structured_query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
)
for row in rows:
    print(f"  {row['builder']} built both")

print("\n=== Who invested in Pied Piper? ===")
rows = graph_store.structured_query(
    "MATCH (i)-[e:INVESTED_IN]->(:Company {name: $company}) "
    "RETURN i.name AS investor, e.round AS round ORDER BY e.date",
    param_map={"company": "Pied Piper"},
)
for row in rows:
    print(f"  {row['investor']}: {row['round']}")

print("\n=== How is Richard connected to Son of Anton? (GQL) ===")
rows = graph_store.structured_query(
    "MATCH p = SHORTEST 1 (r:Person {name: 'Richard Hendricks'})-[*1..4]-(s:AI {name: 'Son of Anton'}) "
    "RETURN [n IN nodes(p) | n.name] AS path",
    language="gql",
)
for row in rows:
    print("  " + " -> ".join(row["path"]))

print("\n=== Russ Hanneman ===")
for row in graph_store.structured_query(
    "MATCH (p:Person {name: 'Russ Hanneman'}) RETURN p.net_worth_commas AS commas, p.roi_stands_for AS roi"
):
    print(f"  commas in his net worth: {row['commas']}; ROI stands for: {row['roi']}")

# ── Vector search, narrowed by a filter ─────────────────────────────────

print("\n=== Nearest to middle-out, with a Weissman score of at least 5 ===")
query = VectorStoreQuery(
    query_embedding=[float(x) for x in vectors.by_key["middle-out"]],
    similarity_top_k=3,
    filters=MetadataFilters(filters=[MetadataFilter(key="weissman_score", value=5, operator=FilterOperator.GTE)]),
)
found, scores = graph_store.vector_query(query)
for node, score in zip(found, scores, strict=True):
    print(f"  {node.id}: similarity={score:.3f}")

# ── Graph algorithms: who is most central? ──────────────────────────────

print("\n=== PageRank, top 5 ===")
ranks = graph_store.client.algorithms.pagerank()
for node_id, score in sorted(ranks.items(), key=lambda item: item[1], reverse=True)[:5]:
    print(f"  {graph_store.client.get_node(node_id).properties()['name']}: {score:.4f}")

print("\n=== Schema ===")
print(graph_store.get_schema_str())
graph_store.close()
