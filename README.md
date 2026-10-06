[![CI](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex)
[![PyPI](https://img.shields.io/pypi/v/grafeo-llamaindex.svg)](https://pypi.org/project/grafeo-llamaindex/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

# grafeo-llamaindex

LlamaIndex `PropertyGraphStore` backed by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

Build knowledge graphs from documents, query them with GQL, and run vector similarity search, all in a single `.grafeo` file. No servers, no infrastructure.

## Install

```bash
uv add grafeo-llamaindex
```

Requires Python 3.12+ and grafeo `>=0.5.44,<0.6`, which is installed with it. Support for grafeo 0.6 comes with grafeo-llamaindex 0.3.0.

## Quickstart

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from grafeo_llamaindex import GrafeoPropertyGraphStore

documents = SimpleDirectoryReader("./data").load_data()

graph_store = GrafeoPropertyGraphStore(db_path="./knowledge_graph.grafeo")

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,
)

retriever = index.as_retriever(include_text=True)
nodes = retriever.retrieve("What are the key relationships?")
```

## Features

- **Full PropertyGraphStore**: all 8 abstract methods implemented (`get`, `get_triplets`, `get_rel_map`, `upsert_nodes`, `upsert_relations`, `delete`, `structured_query`, `vector_query`)
- **Structured + vector queries**: `supports_structured_queries = True` and `supports_vector_queries = True` in a single store
- **Embedded database**: no Docker, no cloud, no external services. Just `uv add grafeo`
- **Single-file persistence**: the entire knowledge graph lives in one `.grafeo` file
- **Native HNSW vector search**: embeddings stored alongside graph nodes, no separate vector DB needed
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ all supported
- **Built-in graph algorithms**: PageRank, Louvain, shortest paths, centrality and 30+ more via `graph_store.client.algorithms`

## API Reference

### `GrafeoPropertyGraphStore`

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

store = GrafeoPropertyGraphStore(
    db_path=None,                # str | None - path for persistent storage, None for in-memory
    embedding_dimensions=1536,   # int - vector dimensions; must match your embedding model
    embedding_metric="cosine",   # str - "cosine", "euclidean", "dot_product", or "manhattan"
    dedup_threshold=None,        # float | None - cosine similarity threshold for entity dedup
    query_language="gql",        # str - "gql" or "cypher": the language of structured_query
)
```

**Properties:**

- `store.client`: access the underlying `grafeo.GrafeoDB` instance for direct queries and algorithms
- `store.supports_structured_queries`: `True`
- `store.supports_vector_queries`: `True`

**Methods (PropertyGraphStore interface):**

| Method | Description |
| --- | --- |
| `upsert_nodes(nodes)` | Insert or update `EntityNode` / `ChunkNode` objects (`ValueError` if an embedding does not have `embedding_dimensions` values) |
| `upsert_relations(relations)` | Insert edges between existing nodes |
| `get(properties, ids)` | Retrieve nodes by ID or property filter |
| `get_triplets(entity_names, relation_names, properties, ids, limit)` | Get `(source, relation, target)` triplets |
| `get_rel_map(graph_nodes, depth, limit, ignore_rels)` | BFS from seed nodes, following relations in both directions |
| `delete(entity_names, relation_names, properties, ids)` | Remove nodes (with all their relations) and/or relations |
| `structured_query(query, param_map, language)` | Execute a raw query in `language` (default: the store's `query_language`), or Gremlin when it starts with `g.`; `param_map` values are passed as query parameters (`$name`) |
| `vector_query(query)` | HNSW similarity search over node embeddings |
| `get_schema()` / `get_schema_str()` | Inspect graph labels, edge types, and properties |
| `persist(path)` | Save an in-memory database to disk (use a `.grafeo` path) |
| `close()` | Close the database connection |

## Persistence

The entire knowledge graph lives in a single `.grafeo` file. Pass `db_path` to store data on disk, or omit it for in-memory use.

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

# Create and populate
store = GrafeoPropertyGraphStore(db_path="./my_graph.grafeo")
# ... upsert nodes and relations ...
store.close()

# Reopen later with the same path
store = GrafeoPropertyGraphStore(db_path="./my_graph.grafeo")
print(store.node_count, store.edge_count)  # data is still there
```

You can also save an in-memory store to disk:

```python
store = GrafeoPropertyGraphStore()  # in-memory
# ... populate ...
store.persist("./snapshot.grafeo")
```

Use the `.grafeo` extension. Grafeo stores any other path as a directory with a write-ahead log, a layout that is deprecated from grafeo 0.6.

Reopen a store with the same `embedding_dimensions` it was written with. Vector indexes are rebuilt on open; a label whose stored embeddings have other dimensions is reported with a `UserWarning` and left out of `vector_query`.

### Upgrading to grafeo 0.6

grafeo 0.6 writes a new file format. The first time 0.6 opens a database written by 0.5.x for writing, it migrates the file and keeps the old one next to it (`my_graph.grafeo.pre-0.6`). After that, grafeo 0.5.x can no longer open the database, so close it in every 0.5.x process first and keep a backup if you may need to go back. See [Upgrading from 0.5](https://grafeo.dev/user-guide/persistence/persistent/#upgrading-from-05).

## Deduplication

When `dedup_threshold` is set, `upsert_nodes` checks whether an incoming `EntityNode`'s embedding is similar enough to an existing node (same label) to merge them instead of creating a duplicate.

```python
store = GrafeoPropertyGraphStore(
    dedup_threshold=0.95,  # cosine similarity threshold
    embedding_dimensions=1536,
)
```

Key behavior:

- **Threshold semantics**: if `cosine_similarity(new, existing) >= dedup_threshold`, the new node merges into the existing one (properties are overwritten, the original `created_at` timestamp is preserved).
- **Label-scoped**: dedup only compares nodes with the same label. A "Person" and a "Company" with identical embeddings are never merged.
- **ChunkNode excluded**: `ChunkNode` objects are never deduplicated, only `EntityNode`.
- **Requires embedding**: nodes without an embedding are never deduplicated.
- **Runtime toggle**: you can set `store.dedup_threshold = 0.9` at any time and it takes effect on the next `upsert_nodes` call.

## Relation Upsert Behavior

Relation labels are stored as edge types in upper case, with spaces and hyphens turned into underscores (`"works at"` becomes `WORKS_AT`); any other character is kept. `get_triplets(relation_names=...)` and `delete(relation_names=...)` accept a label as given or as stored. Upserting a relation that already exists between the same two nodes updates its properties instead of adding a second edge.

`upsert_relations` skips relations whose `source_id` or `target_id` does not match any existing node (by name or LlamaIndex ID). A `UserWarning` is emitted for each skipped relation, so you can catch these with Python's `warnings` module if needed.

## Structured Queries

`structured_query` runs GQL by default. Pass `query_language="cypher"` to run Cypher instead, or choose per call with `structured_query(query, language="cypher")` (also on `astructured_query`). A query that starts with `g.` always runs as Gremlin.

LlamaIndex's `TextToCypherRetriever` and `CypherTemplateRetriever` generate Cypher, so create the store with `query_language="cypher"` when you use them:

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

store = GrafeoPropertyGraphStore(db_path="./my_graph.grafeo", query_language="cypher")
retriever = TextToCypherRetriever(store, llm=llm)
```

GQL and Cypher share most read queries, but Cypher-only syntax, such as a pattern in `WHERE` or `=~`, fails as GQL.

## Graph Traversal

`get_rel_map` follows relations in both directions, like LlamaIndex's reference stores: from each seed node it collects the relations up to `depth` hops away, each once and in its stored direction, nearest hops first, at most `limit` of them. Relations whose label is in `ignore_rels` are neither returned nor followed.

`GrafeoPGRetriever` blends vector similarity with PageRank. The PageRank scores come from Grafeo and are cached until the graph changes: writes through the store clear the cache, and writes made directly through `store.client` are detected by the database epoch.

## Deleting

`delete(ids=...)`, `delete(entity_names=...)` and `delete(properties=...)` remove every matching node together with all relations attached to it, in either direction. `delete(relation_names=...)` removes relations only.

## Threads

A store can be shared between threads (LlamaIndex uses threads for `num_workers` and async retrieval). Writes (`upsert_nodes`, `upsert_relations`, `delete`) run one at a time, so concurrent upserts of the same node create it once.

## Comparison

| | Neo4j | FalkorDB | **Grafeo** |
| --- | --- | --- | --- |
| Requires server | Yes | Yes | **No** (embedded) |
| Vector search | Plugin (5.x+) | Limited | **Native HNSW** |
| Graph algorithms | GDS plugin ($) | Built-in | **Built-in (30+)** |
| Query languages | Cypher | Cypher | **GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ** |
| Deployment | Docker/Cloud | Docker/Cloud | **`uv add grafeo`** |
| Persistence | Server-managed | Server-managed | **Single `.grafeo` file** |

## Examples

See the [`examples/`](examples/) directory:

- **[`mock_embedding_demo.py`](examples/mock_embedding_demo.py)**: full demo with hand-crafted embeddings, no API key required
- **[`basic_graph_rag.py`](examples/basic_graph_rag.py)**: build a Property Graph Index from documents and query it (requires OpenAI API key)
- **[`hybrid_retrieval.py`](examples/hybrid_retrieval.py)**: structured queries + vector search + PageRank, all in one script

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
