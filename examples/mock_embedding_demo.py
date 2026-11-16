"""Silicon Valley graph demo with the dataset's own embeddings (no API key or model required).

Loads the Silicon Valley (Pied Piper) dataset from grafeo.ai into a GrafeoPropertyGraphStore,
with the 384-dimension MiniLM vectors the dataset ships, then runs vector search, graph
traversal and lookups.

Usage:
    uv add grafeo-llamaindex "grafeo-datasets[grafeo]"
    uv run python examples/mock_embedding_demo.py
"""

from __future__ import annotations

from grafeo_datasets import silicon_valley
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

vectors = silicon_valley.vectors()  # MiniLM vectors of each node's description
store = GrafeoPropertyGraphStore(embedding_dimensions=vectors.dimension)

nodes = silicon_valley.nodes()
names = {n["key"]: n["name"] for n in nodes}
store.upsert_nodes(
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
store.upsert_relations(
    [
        Relation(label=e["type"], source_id=names[e["source"]], target_id=names[e["target"]], properties=e["props"])
        for e in silicon_valley.edges()
    ]
)
print(store.summary())

# Vector search: what is most like Jian-Yang's fridge?
fridge = [float(x) for x in vectors.by_key["jian-yangs-fridge"]]
found, scores = store.vector_query(VectorStoreQuery(query_embedding=fridge, similarity_top_k=4))
print("\n=== Closest to Jian-Yang's fridge ===")
for node, score in zip(found, scores, strict=True):
    print(f"  {node.id} ({node.label}): similarity={score:.3f}")

[fridge_node] = store.get(ids=["Jian-Yang's fridge"])
print(f"\nDoes the fridge talk? {fridge_node.properties['talks']}")

# Graph traversal: everything Gilfoyle built, and who rivals whom
print("\n=== What Gilfoyle built ===")
for src, rel, tgt in store.get_triplets(entity_names=["Bertram Gilfoyle"], relation_names=["BUILT"]):
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

print("\n=== Rivals ===")
for src, rel, tgt in store.get_triplets(relation_names=["RIVAL_OF"]):
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

print("\n=== Around Son of Anton, two hops ===")
for src, rel, tgt in store.neighbors("Son of Anton", depth=2, limit=8):
    print(f"  {src.id} --[{rel.label}]--> {tgt.id}")

[anton_son] = store.get(ids=["Son of Anton"])
print(f"\nSon of Anton fixes bugs by: {anton_son.properties['bug_fix_strategy']}")
store.close()
