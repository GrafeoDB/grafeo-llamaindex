<picture>
  <source media="(max-width: 556px) and (prefers-color-scheme: dark), (min-width: 768px) and (max-width: 844px) and (prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-llamaindex/main/.github/readme/header-narrow-dark.svg">
  <source media="(max-width: 556px), (min-width: 768px) and (max-width: 844px)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-llamaindex/main/.github/readme/header-narrow-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-llamaindex/main/.github/readme/header-dark.svg">
  <img alt="grafeo-llamaindex: A LlamaIndex property graph store backed by Grafeo, in one file." src="https://raw.githubusercontent.com/GrafeoDB/grafeo-llamaindex/main/.github/readme/header-light.svg" width="100%">
</picture>

[![CI](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-llamaindex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-llamaindex)
[![PyPI](https://img.shields.io/pypi/v/grafeo-llamaindex.svg)](https://pypi.org/project/grafeo-llamaindex/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/LICENSE)

LlamaIndex `PropertyGraphStore` backed by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

Build knowledge graphs from documents, query them with GQL, and run vector similarity search, all in a single `.grafeo` file. No servers, no infrastructure.

## Install

```bash
uv add grafeo-llamaindex
```

Requires Python 3.12+, `llama-index-core` 0.14 or later, and grafeo 0.6 (`grafeo>=0.6.2,<0.7`), which is installed with it. For grafeo 0.5, use grafeo-llamaindex 0.2.0 and pin `grafeo<0.6` as well: 0.2.0 requires `grafeo>=0.5` without an upper bound, so a new install of it gets grafeo 0.6. A database written with grafeo 0.5 is migrated the first time grafeo 0.6 opens it: read [Upgrading from grafeo 0.5](#upgrading-from-grafeo-05) first.

## Quickstart

The quickstart loads the Silicon Valley (Pied Piper) dataset from [grafeo.ai](https://grafeo.ai) and asks Mistral about it (other models below):

```bash
uv add grafeo-llamaindex "grafeo-datasets[grafeo]" llama-index-llms-mistralai llama-index-embeddings-mistralai
export MISTRAL_API_KEY=...
```

```python
from grafeo_datasets import silicon_valley
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from grafeo_llamaindex import GrafeoPropertyGraphStore

Settings.llm = MistralAI(model="mistral-small-latest")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")

graph_store = GrafeoPropertyGraphStore(
    db_path="./pied_piper.grafeo",
    embed_model=Settings.embed_model,  # reads 1024 dimensions from mistral-embed
)

# Pied Piper, its people, products and rivals: 64 entities and their relations
nodes = silicon_valley.nodes()
names = {n["key"]: n["name"] for n in nodes}
embeddings = Settings.embed_model.get_text_embedding_batch([f"{n['name']}: {n['description']}" for n in nodes])
graph_store.upsert_nodes(
    [
        EntityNode(
            name=n["name"], label=n["label"], properties={"description": n["description"], **n["props"]}, embedding=e
        )
        for n, e in zip(nodes, embeddings, strict=True)
    ]
)
graph_store.upsert_relations(
    [
        Relation(label=e["type"], source_id=names[e["source"]], target_id=names[e["target"]], properties=e["props"])
        for e in silicon_valley.edges()
    ]
)

index = PropertyGraphIndex.from_existing(property_graph_store=graph_store, use_async=False)
query_engine = index.as_query_engine(include_properties=True)
print(query_engine.query("Does Jian-Yang's fridge talk?"))
# for example: Yes, Jian-Yang's fridge has a screen and a voice.
```

The store keeps 147 relations of the dataset's 149 edges: a relation upsert updates the relation with the same source, label and target, so Richard Hendricks's three `WORKS_AT` edges to Pied Piper, one per role, become one relation that holds the last role.

Use a model alias such as `mistral-small-latest`: `llama-index-llms-mistralai` 0.12 checks model names against a fixed table and rejects dated IDs such as `mistral-small-2603` with `ValueError: Unknown model`. Without `Settings.llm` and `Settings.embed_model`, LlamaIndex falls back to OpenAI's models.

The graph is saved as you go. Reopen it later with the same embedding model:

```python
graph_store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo", embed_model=Settings.embed_model)
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store, embed_model=Settings.embed_model, use_async=False
)
```

`embed_model` makes the store embed one short text when it is created, to read the size of the model's embeddings. Without it, pass `embedding_dimensions` (default 1536). To try the store without an API key, run [`examples/mock_embedding_demo.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/mock_embedding_demo.py), or use a local model (below).

### Building from your own documents

`PropertyGraphIndex` extracts entities and relations from documents with the LLM:

```python
from llama_index.core import SimpleDirectoryReader

index = PropertyGraphIndex.from_documents(
    SimpleDirectoryReader("./data").load_data(),
    property_graph_store=graph_store,
    embed_kg_nodes=True,
    use_async=False,  # see the note below
)
```

**Why `use_async=False`**: with LlamaIndex's default `use_async=True`, `PropertyGraphIndex` embeds the chunks and the extracted entities in two separate `asyncio.run()` calls on the same embedding client. API clients built on `httpx` (Mistral, Anthropic, OpenAI, Ollama) keep a pooled connection from the first event loop, and the second call fails with `RuntimeError: Event loop is closed`, on Windows and Linux alike. This is a LlamaIndex issue that any graph store meets; `use_async=False` avoids it, and entity extraction still runs its LLM calls concurrently.

### Adding documents later

Entity extraction runs in its own `asyncio.run()` on every build and insert, whatever `use_async` is, so `index.insert(document)` with the LLM object the index was built with fails with `Event loop is closed` (the same LlamaIndex issue). An index keeps the extractors it was built with, so setting `Settings.llm` again does not help. Insert through an index over the same store with a new LLM object instead:

```python
from llama_index.core import Document
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor


def inserting_index(graph_store):
    """An index over graph_store whose extractor has a new LLM client."""
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        kg_extractors=[SimpleLLMPathExtractor(llm=MistralAI(model="mistral-small-latest"))],
        embed_model=Settings.embed_model,
        embed_kg_nodes=True,
        use_async=False,
    )


inserting_index(graph_store).insert(Document(text="Jack Barker took over from Gavin Belson as CEO of Hooli."))

for source, relation, target in graph_store.get_triplets(
    entity_names=["Hooli"], relation_names=["WAS_CEO_OF", "BECAME_CEO_OF"]
):
    print(source.id, relation.label, target.id)
# for example: Gavin Belson WAS_CEO_OF Hooli, Jack Barker BECAME_CEO_OF Hooli
```

The extracted "Gavin belson" and "Jack barker" (LlamaIndex's extractor capitalises names) join the dataset's Gavin Belson and Jack Barker, since names match ignoring case (`case_insensitive_names`). Call `inserting_index()` again for each later insert.

### Other models

Set `Settings.llm` and `Settings.embed_model` before creating the store; the rest of the quickstart stays the same.

**A local model with Ollama** (no API key; run `ollama pull ministral-3:3b` and `ollama pull nomic-embed-text` first):

```bash
uv add grafeo-llamaindex llama-index-llms-ollama llama-index-embeddings-ollama
```

```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="ministral-3:3b", request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

**Anthropic**, which has no embedding model, so it is paired with Mistral's here:

```bash
uv add grafeo-llamaindex llama-index-llms-anthropic llama-index-embeddings-mistralai
```

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.anthropic import Anthropic

Settings.llm = Anthropic(model="claude-opus-5-5")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
```

**OpenAI**:

```bash
uv add grafeo-llamaindex llama-index-llms-openai llama-index-embeddings-openai
```

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

### Troubleshooting

- **The graph has only `text_chunk` nodes**: entity extraction failed for every chunk. LlamaIndex's extractors log the error and carry on; pass `kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, raise_on_error=True)]` (from `llama_index.core.indices.property_graph`) to `from_documents` to see it (a rejected model name, a missing API key, a rate limit).
- **`RuntimeError: Event loop is closed`**: while building an index, pass `use_async=False`; on a later `insert`, insert through a new index with a new LLM object (see [Adding documents later](#adding-documents-later)).
- **`ValueError` about embedding dimensions**: the store was created with other dimensions than the embedding model gives. Pass `embed_model=` to the store, or the model's size as `embedding_dimensions`.

## Features

- **Full PropertyGraphStore**: all 8 abstract methods implemented (`get`, `get_triplets`, `get_rel_map`, `upsert_nodes`, `upsert_relations`, `delete`, `structured_query`, `vector_query`)
- **Structured + vector queries**: `supports_structured_queries = True` and `supports_vector_queries = True` in a single store
- **Embedded database**: no Docker, no cloud, no external services. Just `uv add grafeo-llamaindex`
- **Single-file persistence**: the entire knowledge graph lives in one `.grafeo` file
- **Native HNSW vector search**: embeddings stored alongside graph nodes, no separate vector DB needed
- **Multi-language queries**: `structured_query` runs Cypher, GQL and Gremlin; GraphQL, SPARQL and SQL/PGQ run through `store.client` (`execute_graphql`, `execute_sparql`, `execute_sql`)
- **Built-in graph algorithms**: PageRank, Louvain, shortest paths, centrality and 30+ more via `graph_store.client.algorithms`

## API Reference

### `GrafeoPropertyGraphStore`

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

store = GrafeoPropertyGraphStore(
    db_path=None,  # str | PathLike | None: path for persistent storage, None for in-memory
    embed_model=None,  # BaseEmbedding | None: read embedding_dimensions from this model
    embedding_dimensions=None,  # int | None: vector dimensions; 1536 without embed_model
    embedding_metric="cosine",  # str: "cosine", "euclidean", "dot_product", or "manhattan"
    dedup_threshold=None,  # float | None: cosine similarity threshold for entity dedup
    query_language="cypher",  # str: "cypher" or "gql", the language of structured_query
    case_insensitive_names=True,  # bool: match entity names ignoring case on upsert and lookup
)
```

With both `embed_model` and `embedding_dimensions`, they must agree, or the store raises `ValueError` before it opens the database.

With `case_insensitive_names` (the default), an entity whose name matches an existing one ignoring case merges into it, as relation endpoints and `get_triplets` / `delete` by name also match. LlamaIndex's default extractor capitalises names (`"Gavin Belson"` becomes `"Gavin belson"`), so extracted entities join the ones you wrote yourself instead of duplicating them. The existing entity keeps its name and label, and the other spelling is kept as an alias. Pass `case_insensitive_names=False` to keep names that differ only in case apart.

**Properties:**

- `store.client`: access the underlying `grafeo.GrafeoDB` instance for direct queries and algorithms
- `store.query_language`: the language `structured_query` uses when a call does not name one
- `store.embedding_dimensions`: the size every embedding and vector query must have
- `store.node_count` / `store.edge_count`: number of nodes and relations in the graph
- `store.dedup_threshold`: the dedup threshold, settable at any time (see [Deduplication](#deduplication))
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
| `vector_query(query)` | HNSW similarity search over entity embeddings (not chunks), narrowed by `query.filters` and `query.node_ids` (see [Vector Search](#vector-search)) |
| `get_schema()` / `get_schema_str()` | Inspect graph labels, edge types, and properties |
| `persist(path)` | Save an in-memory database to a file (use a `.grafeo` path), replacing an earlier copy |
| `close()` | Close the database connection |

**Convenience methods:**

| Method | Description |
| --- | --- |
| `neighbors(name_or_id, depth=1, limit=30)` | Triplets around one node, found by LlamaIndex ID or entity name |
| `summary()` | Readable overview: node and edge counts per label and type, dedup setting, persistence |
| `with GrafeoPropertyGraphStore(...) as store:` | Closes the store when the block ends |

### `GrafeoPGRetriever`

A sub-retriever for `PropertyGraphIndex` that finds seed nodes by vector search, expands them through the graph, and ranks the triplets by `alpha * relevance + (1 - alpha) * pagerank`. The relevance of a triplet is the mean similarity of the query to its source, its relation label and its target, so the triplet that answers ranks above triplets of well-connected hubs; PageRank, computed by Grafeo, adds a smaller weight. `limit` triplets are returned, ranked among `4 * limit` candidates.

```python
from grafeo_llamaindex import GrafeoPGRetriever

retriever = index.as_retriever(
    sub_retrievers=[
        GrafeoPGRetriever(
            graph_store,
            embed_model=None,  # BaseEmbedding | None: None uses Settings.embed_model
            similarity_top_k=10,  # int: seed nodes from vector search
            path_depth=1,  # int: hops to expand from each seed
            limit=30,  # int: maximum triplets returned
            alpha=0.9,  # float: weight of relevance against PageRank
            use_pagerank=True,  # bool: False ranks by relevance only
            include_text=True,  # bool: include the source text of each triplet
            include_properties=False,  # bool: include node and relation properties in the text
            # **kwargs go to LlamaIndex's BasePGRetriever
        )
    ]
)
nodes = retriever.retrieve("What did Gilfoyle build?")
# for example: Bertram Gilfoyle -> BUILT -> Anton, Bertram Gilfoyle -> BUILT -> Son of Anton, ...
```

## Persistence

The entire knowledge graph lives in a single `.grafeo` file. Pass `db_path` to store data on disk, or omit it for in-memory use.

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

# Create and populate
store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo")
# ... upsert nodes and relations ...
store.close()

# Reopen later with the same path
store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo")
print(store.node_count, store.edge_count)  # data is still there
```

You can also save an in-memory store to disk:

```python
store = GrafeoPropertyGraphStore()  # in-memory
# ... populate ...
store.persist("./snapshot.grafeo")
```

A database is always a single file, whatever its extension; use `.grafeo`. While the store is open, grafeo keeps its write-ahead log in a directory next to it (`pied_piper.grafeo.wal/`) and folds it into the file on `close()`. `db_path` must be a file path: a directory fails with `grafeo.GrafeoError`, an empty one included, unless it is a database written by grafeo 0.5 (see below). `persist()` replaces a file that is already at its path.

Reopen a store with the same `embedding_dimensions` it was written with. Vector indexes are rebuilt on open; a label whose stored embeddings have other dimensions is reported with a `UserWarning` and left out of `vector_query`.

### With LlamaIndex's StorageContext

`StorageContext.persist(persist_dir)` saves an in-memory store as `<persist_dir>/property_graph_store.grafeo`: `persist()` writes a path ending in `.json`, the name LlamaIndex passes, with `.grafeo` instead. To load the index again, open that file and pass the store to the storage context:

```python
from llama_index.core import StorageContext, load_index_from_storage

graph_store = GrafeoPropertyGraphStore(db_path="./storage/property_graph_store.grafeo")
storage_context = StorageContext.from_defaults(persist_dir="./storage", property_graph_store=graph_store)
index = load_index_from_storage(storage_context)
```

A store opened with `db_path` writes to its own file as it goes, and `persist()` leaves it as it is.

### Upgrading from grafeo 0.5

grafeo-llamaindex 0.3 requires grafeo 0.6, which stores databases in a new format. Databases written by grafeo-llamaindex 0.2 on grafeo 0.5 keep working, after a one-way migration:

1. **Back up** the database first: copy the `.grafeo` file, or the whole directory if grafeo 0.5 wrote a WAL directory (0.5 did that for any path without the `.grafeo` extension).
2. **Stop every process** that opens it with grafeo 0.5. grafeo 0.5 cannot open the migrated database.
3. **Open it once** with grafeo 0.6, for example with `GrafeoPropertyGraphStore(db_path=...)`. The first read-write open migrates it in place: the path becomes a single 0.6 database file, and the old database is kept, unchanged, next to it as `<path>.pre-0.6` (a file, or a directory for a WAL directory). Nodes, relations, properties and embeddings carry over, and the store rebuilds its vector indexes on open.
4. **Keep the `.pre-0.6` copy** until you no longer need to go back. To return to grafeo 0.5, stop every process that opens the database, delete the migrated file (and the `<path>.wal` directory next to it, if there is one), rename the `.pre-0.6` copy to the original path, and run `uv add "grafeo-llamaindex==0.2.0" "grafeo<0.6"`: grafeo-llamaindex 0.2.0 does not cap grafeo, so without the second pin it would install grafeo 0.6 and migrate the database again. grafeo 0.7 will no longer read 0.5 databases, so migrate them with 0.6.

What else changes with grafeo 0.6:

- A property set to `None` is not stored, and properties grafeo 0.5 stored as null are gone after the migration.
- `decimal.Decimal` properties keep their exact value and scale, and in GQL a number with a point (`1.50`) is an exact decimal, so GQL results can hold `Decimal` values. Cypher `1.5` stays a float.
- GQL `ORDER BY` sorts nulls last in both directions (Cypher sorts them first when descending). Write `ORDER BY x DESC NULLS FIRST` for the descending order of grafeo 0.5.
- Integer and decimal overflow and division by zero fail the query instead of giving null, and in GQL a `WHERE` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is an error.

See [Upgrading from 0.5](https://docs.grafeo.dev/operate/persistence/persistent/#upgrading-from-05) and [Upgrading to 0.6](https://docs.grafeo.dev/get-started/upgrading/0.6/) in the grafeo documentation.

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
- **The existing node keeps its identity**: it keeps its ID and name, and the merged node's ID is saved as an alias, so `get(ids=...)`, `get_triplets(entity_names=..., ids=...)`, `delete(ids=...)` and relations that name it find the merged node, also after a reopen.
- **Label-scoped**: dedup only compares nodes with the same label. A "Person" and a "Company" with identical embeddings are never merged.
- **ChunkNode excluded**: `ChunkNode` objects are never deduplicated, only `EntityNode`.
- **Requires embedding**: nodes without an embedding are never deduplicated.
- **Runtime toggle**: you can set `store.dedup_threshold = 0.9` at any time and it takes effect on the next `upsert_nodes` call.

## Vector Search

`vector_query` searches the embeddings of entity nodes only. LlamaIndex embeds text chunks too, but like its Neo4j store, chunks are left out so they do not take the seed slots of `VectorContextRetriever`. A chunk's embedding is still stored and returned by `get()`. Chunks are recognized by their kind, not their label, so a `ChunkNode` with its own label is left out too, also on a label it shares with entities.

`query.filters` and `query.node_ids` narrow the search inside the HNSW index, so `similarity_top_k` results still come back when they exist:

```python
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=Settings.embed_model.get_query_embedding("a kitchen appliance that talks"), similarity_top_k=3
)
nodes, scores = graph_store.vector_query(query)  # for example: Jian-Yang's fridge first

commas = MetadataFilters(filters=[MetadataFilter(key="net_worth_commas", value=2, operator=FilterOperator.GTE)])
retriever = index.as_retriever(filters=commas)  # passed on to VectorContextRetriever: only Russ Hanneman qualifies
```

Filters combine with AND, with the operators `==`, `!=`, `>`, `>=`, `<`, `<=`, `in` and `nin`. Filters combined with OR or NOT, nested filters and other operators raise `NotImplementedError`.

## Node Upsert Behavior

Upserting a node that already exists updates its properties; properties it does not name are kept. When its label changed, for example when an extractor labels the same entity `PERSON` in one chunk and `CHARACTER` in another, the node moves to the new label, in `get()`, the schema and `MATCH (n:CHARACTER)` alike. Property keys may be any text without a backtick, such as LlamaIndex's `file name` metadata, also in `get(properties=...)` and `delete(properties=...)`.

## Relation Upsert Behavior

Relation labels are stored as edge types in upper case, with spaces and hyphens turned into underscores (`"works at"` becomes `WORKS_AT`); any other character is kept. `get_triplets(relation_names=...)` and `delete(relation_names=...)` accept a label as given or as stored. Upserting a relation that already exists between the same two nodes updates its properties instead of adding a second edge.

`upsert_relations` skips relations whose `source_id` or `target_id` does not match any existing node (by name or LlamaIndex ID). A `UserWarning` is emitted for each skipped relation, so you can catch these with Python's `warnings` module if needed. The one exception is LlamaIndex's `SOURCE` relation from a chunk to its document, which `ImplicitPathExtractor` (one of `PropertyGraphIndex`'s default extractors) adds for every chunk: documents are not graph nodes, so it is skipped without a warning, and the chunk keeps its document ID as `ref_doc_id`.

## Structured Queries

`structured_query` runs Cypher by default (GQL before 0.3.0). Pass `query_language="gql"` to run GQL instead, or choose per call with `structured_query(query, language="gql")` (also on `astructured_query`). A query that starts with `g.` always runs as Gremlin.

`get_schema_str()` describes the graph for an LLM that writes queries, in the layout of LlamaIndex's Neo4j store:

```text
Node properties:
- AI: bug_fix_strategy, created_at, description, form, meat_ordered_for_lunch_lb, name, talks, updated_at
- Company: created_at, description, founded, motto, name, name_bought_from, rejected_offer_usd, updated_at
...
Relationship properties:
- INVESTED_IN: amount_usd, created_at, date, note, round, updated_at
...
The relationships:
(:Person)-[:BUILT]->(:AI)
(:Person)-[:INVESTED_IN]->(:Company)
...
```

(An excerpt for the quickstart's Silicon Valley graph.)

It leaves out the keys the store keeps for itself (`li_id`, `li_label`, `li_kind`, `li_sources`, `li_aliases`, `embedding`) and LlamaIndex's private ones (`triplet_source_id`, keys starting with `_`). Queries through `structured_query()` and `client` still see them.

In an index built from files, the lists are longer: LlamaIndex's extractors copy each chunk's metadata onto the entities and relations extracted from it, so `SimpleDirectoryReader`'s `file_name`, `file_path`, `file_type`, `file_size`, `creation_date` and `last_modified_date` show up on chunks, entities and relations alike. They are real properties an LLM can query (`WHERE e.file_name = 'report.pdf'`), so they stay in the schema.

Queries against the quickstart's graph, with their results:

```python
graph_store.structured_query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
)
# [{'builder': 'Bertram Gilfoyle'}]: Gilfoyle built both

graph_store.structured_query(
    "MATCH (i)-[e:INVESTED_IN]->(:Company {name: $company}) RETURN i.name AS investor, e.round AS round ORDER BY e.date",
    param_map={"company": "Pied Piper"},
)
# Erlich Bachman (Incubator stake), Peter Gregory (Seed), Russ Hanneman (Series A), Raviga Capital (Secondary)

graph_store.structured_query("MATCH (s:AI {name: 'Son of Anton'}) RETURN s.bug_fix_strategy AS strategy")
# [{'strategy': 'delete the code'}]
```

LlamaIndex's `TextToCypherRetriever` and `CypherTemplateRetriever` generate Cypher, which is the default:

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

retriever = TextToCypherRetriever(graph_store, llm=Settings.llm)  # or any LlamaIndex LLM
nodes = retriever.retrieve("Who invested in Pied Piper?")
```

For text-to-Cypher, build the graph with `SchemaLLMPathExtractor` (typed entity labels and allowed relations) rather than the default `SimpleLLMPathExtractor`, which labels every entity `entity`. The schema text then shows which labels each relation connects, such as `(:COMPANY)-[:BASED_IN]->(:CITY)`, so the model writes queries against the real structure.

GQL and Cypher share most read queries, but Cypher-only syntax, such as a pattern in `WHERE`, fails as GQL. They also differ in values: in GQL a number with a point is an exact decimal (`decimal.Decimal` in the results) and `ORDER BY` puts nulls last, while Cypher reads `1.5` as a float and puts nulls first when sorting descending.

## Graph Traversal

`get_rel_map` follows relations in both directions, like LlamaIndex's reference stores: from each seed node it collects the relations up to `depth` hops away, each once and in its stored direction, nearest hops first, at most `limit` of them. Relations whose label is in `ignore_rels` are neither returned nor followed.

`GrafeoPGRetriever` blends the relevance of each triplet with PageRank. The PageRank scores come from Grafeo and are cached until the graph changes: writes through the store clear the cache, and writes made directly through `store.client` are detected by the database epoch.

## Deleting

`delete(ids=...)`, `delete(entity_names=...)` and `delete(properties=...)` remove every matching node together with all relations attached to it, in either direction. `delete(relation_names=...)` removes relations only.

Deleting a chunk also deletes what was extracted only from it, so `index.delete_ref_doc(doc_id)` removes a document's knowledge and keeps what other documents also mention. Entities and relations record the chunks they were extracted from (LlamaIndex's `triplet_source_id`, collected in `li_sources`):

- A relation goes when all its source chunks are deleted; otherwise its `triplet_source_id` moves to a remaining chunk.
- An entity goes when all its source chunks are deleted and no relation holds it any more. An entity first extracted from a deleted chunk stays while a relation from another chunk names it.
- Entities and relations written without a `triplet_source_id`, such as ones you upsert yourself, are never deleted this way.
- Data written by grafeo-llamaindex 0.2 has no `li_sources`; its `triplet_source_id` counts as its one source.

`index.delete_ref_doc(doc_id)` finds the document's chunks in the index's docstore. An index loaded with `PropertyGraphIndex.from_existing(...)` starts with an empty docstore, unless it was persisted and reloaded through a `StorageContext` (see [With LlamaIndex's StorageContext](#with-llamaindexs-storagecontext)), so `delete_ref_doc` then deletes nothing. Delete through the store instead, which finds the chunks by their `ref_doc_id` in the graph and cascades the same way:

```python
graph_store.delete_llama_nodes(ref_doc_ids=[doc_id])
```

## Threads

A store can be shared between threads (LlamaIndex uses threads for `num_workers` and async retrieval). Writes (`upsert_nodes`, `upsert_relations`, `delete`) run one at a time, so concurrent upserts of the same node create it once.

## Examples

See the [`examples/`](https://github.com/GrafeoDB/grafeo-llamaindex/tree/main/examples) directory:

- **[`mock_embedding_demo.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/mock_embedding_demo.py)**: the Silicon Valley graph with the dataset's own MiniLM embeddings: vector search, traversal and lookups, no API key or model required
- **[`basic_graph_rag.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/basic_graph_rag.py)**: the Silicon Valley graph with Mistral embeddings, questions through a query engine, then a document of your own inserted (requires a Mistral API key)
- **[`hybrid_retrieval.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/hybrid_retrieval.py)**: Cypher and GQL queries, filtered vector search and PageRank on the Silicon Valley graph, no API key required
- **[`graphrag_migration.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/graphrag_migration.py)**: load the parquet output of a Microsoft GraphRAG indexing run (entities, relationships, communities and their reports, from GraphRAG 1.0 and later or earlier versions) into a `.grafeo` store (requires `pandas` and `pyarrow`)

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
