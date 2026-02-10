[![CI](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex)
[![PyPI](https://img.shields.io/pypi/v/grafeo-llamaindex.svg)](https://pypi.org/project/grafeo-llamaindex/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

# grafeo-llamaindex

LlamaIndex `PropertyGraphStore` backed by [GrafeoDB](https://github.com/GrafeoDB/grafeo) — an embedded graph database with native vector search.

Build knowledge graphs from documents, query them with GQL, and run vector similarity search — all in a single `.db` file. No servers, no infrastructure.

## Install

```bash
pip install grafeo-llamaindex
```

## Quickstart

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from grafeo_llamaindex import GrafeoPropertyGraphStore

documents = SimpleDirectoryReader("./data").load_data()

graph_store = GrafeoPropertyGraphStore(db_path="./knowledge_graph.db")

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,
)

retriever = index.as_retriever(include_text=True)
nodes = retriever.retrieve("What are the key relationships?")
```

## Features

- **Full PropertyGraphStore** — all 8 abstract methods implemented (`get`, `get_triplets`, `get_rel_map`, `upsert_nodes`, `upsert_relations`, `delete`, `structured_query`, `vector_query`)
- **Structured + vector queries** — `supports_structured_queries = True` and `supports_vector_queries = True` in a single store
- **Embedded database** — no Docker, no cloud, no external services. Just `pip install grafeo`
- **Single-file persistence** — your entire knowledge graph lives in one `.db` file
- **Native HNSW vector search** — embeddings stored alongside graph nodes, no separate vector DB needed
- **Multi-language queries** — GQL, Cypher, Gremlin, GraphQL, and SPARQL all supported
- **Built-in graph algorithms** — PageRank, Louvain, shortest paths, centrality, and 30+ more via `graph_store.client.algorithms`

## API Reference

### `GrafeoPropertyGraphStore`

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

store = GrafeoPropertyGraphStore(
    db_path=None,                # str | None — path for persistent storage, None for in-memory
    embedding_dimensions=1536,   # int — vector dimensions for HNSW index
    embedding_metric="cosine",   # str — "cosine", "euclidean", "dot_product", or "manhattan"
)
```

**Properties:**

- `store.client` — access the underlying `grafeo.GrafeoDB` instance for direct queries and algorithms
- `store.supports_structured_queries` — `True`
- `store.supports_vector_queries` — `True`

**Methods (PropertyGraphStore interface):**

| Method | Description |
| --- | --- |
| `upsert_nodes(nodes)` | Insert or update `EntityNode` / `ChunkNode` objects |
| `upsert_relations(relations)` | Insert edges between existing nodes |
| `get(properties, ids)` | Retrieve nodes by ID or property filter |
| `get_triplets(entity_names, relation_names, ids)` | Get `(source, relation, target)` triplets |
| `get_rel_map(graph_nodes, depth, ignore_rels)` | BFS traversal from seed nodes |
| `delete(entity_names, relation_names, ids)` | Remove nodes and/or edges |
| `structured_query(query)` | Execute raw GQL/Cypher (or Gremlin with `g.` prefix) |
| `vector_query(query)` | HNSW similarity search over node embeddings |
| `get_schema()` / `get_schema_str()` | Inspect graph labels, edge types, and properties |
| `persist(path)` | Save in-memory database to disk |
| `close()` | Close the database connection |

## Comparison

| | Neo4j | FalkorDB | **Grafeo** |
| --- | --- | --- | --- |
| Requires server | Yes | Yes | **No** (embedded) |
| Vector search | Plugin (5.x+) | Limited | **Native HNSW** |
| Graph algorithms | GDS plugin ($) | Built-in | **Built-in (30+)** |
| Query languages | Cypher | Cypher | **GQL, Cypher, Gremlin, GraphQL, SPARQL** |
| Deployment | Docker/Cloud | Docker/Cloud | **`pip install grafeo`** |
| Persistence | Server-managed | Server-managed | **Single `.db` file** |

## Examples

See the [`examples/`](examples/) directory:

- **[`basic_graph_rag.py`](examples/basic_graph_rag.py)** — build a Property Graph Index from documents and query it
- **[`hybrid_retrieval.py`](examples/hybrid_retrieval.py)** — structured queries + vector search + PageRank, all in one script

## Development

```bash
uv sync                  # install deps
uv run pytest -v         # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
```

## License

Apache-2.0
