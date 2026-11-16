# Changelog

All notable changes to grafeo-llamaindex, a LlamaIndex property graph store backed by Grafeo.

## [0.3.0] - Unreleased

grafeo 0.6 release: requires grafeo 0.6, which stores every database as a single file in a new format and migrates a database written by grafeo 0.5 on its first open, keeping the old one as `<path>.pre-0.6` (see "Upgrading from grafeo 0.5" in the README). `structured_query()` now runs Cypher by default, deleting a document also deletes what was extracted only from it, and the never-published 0.2.1 fixes below ship in this release. For grafeo 0.5, stay on 0.2.0 and pin `grafeo<0.6` as well, since 0.2.0 sets no upper bound on grafeo.

### Added

- **`embed_model` sets the embedding dimensions**: `GrafeoPropertyGraphStore(embed_model=...)` embeds one short text on creation and uses its size as `embedding_dimensions`, which now defaults to `None` (1536 without a model). A model whose size differs from an explicit `embedding_dimensions` raises `ValueError` before the database is opened.
- **Entity names match ignoring case**: with the new `case_insensitive_names` option (on by default), an upserted entity whose name matches an existing one ignoring case merges into it, keeping the existing name and label, and relation endpoints and lookups by name match the same way. LlamaIndex's default extractor capitalises names (`"Gavin belson"`), which made duplicates of entities written with their real case; pass `case_insensitive_names=False` to keep such names apart.
- **`embedding_dimensions` property**: `store.embedding_dimensions` gives the size every embedding and vector query must have.
- **Type hints are published**: the package ships a `py.typed` marker, so mypy, pyright and ty check code against its annotations instead of treating it as untyped.
- **Databases migrated from grafeo 0.5 are tested**: a `.grafeo` file and a WAL directory written by this store on grafeo 0.5.44 migrate on open with nodes, relations, properties, embeddings, vector search, `get_rel_map` and later writes intact. Their `.pre-0.6` copy is kept unchanged.
- **grafeo 0.6 values and files are tested**: decimals in properties, filters and query results, the default null order of Cypher and GQL, single-file databases, refused directories, and `persist()` called twice, also through `StorageContext.persist()`.
- **The engine contract covers grafeo 0.6**: single-file databases and their WAL sidecar, `save()` refusing an existing path, refused directories, null and decimal properties, the hop query of `get_rel_map`, and a wrong-size vector query raising `grafeo.GrafeoError` (it panicked on 0.5.44).
- **Python 3.15 is supported**: CI runs the suite on Python 3.12 to 3.15, and the package lists 3.12 to 3.15 in its classifiers.

### Changed

Changes in the store first, then grafeo 0.6 behavior that shows through it.

- **Breaking: `structured_query()` runs Cypher by default**: `structured_query()` and `astructured_query()` ran GQL; Cypher is the language LlamaIndex's `TextToCypherRetriever` and `CypherTemplateRetriever` generate. Pass `query_language="gql"` to the store, or `language="gql"` per call, to keep GQL.
- **Breaking: deleting a chunk deletes what was extracted only from it**: `delete()` of a chunk, and so `index.delete_ref_doc()`, also deletes the relations no other chunk supports and the entities left with no source chunk and no relation; it deleted the chunk alone, as LlamaIndex's own stores do. Entities and relations record their source chunks in `li_sources`; data written by 0.2 falls back to `triplet_source_id`, and nodes and relations written without one are never deleted this way.
- **`GrafeoPGRetriever` ranks triplets by their relevance to the query**: a triplet scores the mean similarity of the query to its source, relation label and target, where it scored its best seed endpoint, so well-connected hubs no longer outrank the triplet that answers. `alpha` now defaults to 0.9 (was 0.7), and `limit` triplets are chosen among `4 * limit` candidates; relation labels are embedded once each.
- **`get_schema_str()` describes properties and relationship patterns**: it lists the properties of each node label and relation type and the patterns between labels (`(:person)-[:WORKS_AT]->(:company)`), as LlamaIndex's Neo4j store does, so `TextToCypherRetriever` writes queries against real properties. It listed all property keys together, the store's bookkeeping keys (`li_id`, `embedding`, ...) included, which it now leaves out.
- **README and examples use the Silicon Valley dataset**: the quickstart and `examples/` load the Silicon Valley (Pied Piper) graph from `grafeo-datasets` and use Mistral models first, with local (Ollama), Anthropic and OpenAI alternatives. `examples/mock_embedding_demo.py` and `examples/hybrid_retrieval.py` use the dataset's own MiniLM vectors and need no API key.
- **New bookkeeping keys on stored nodes and relations**: `li_kind` (`"chunk"` or `"entity"`), `li_sources`, `li_aliases` and `li_name_key` (the name in folded case), next to `li_id` and `li_label`. They are left out of the nodes, relations and schema the store returns, but `structured_query()` and `client` see them.
- **`persist()` replaces an earlier copy**: it saves to a new file next to the target and moves it over, so persisting again works as on grafeo 0.5, where grafeo 0.6's `save()` alone refuses an existing path and made `StorageContext.persist()` fail from its second call. Persisting to a directory raises `IsADirectoryError`.
- **`db_path` is always a single database file**: whatever its extension, with the write-ahead log in `<db_path>.wal/` while the store is open. grafeo 0.5 created a WAL directory for a path without the `.grafeo` extension.
- **A directory as `db_path` is refused**: it fails with `grafeo.GrafeoError`, an empty directory included, unless it is a grafeo 0.5 database (grafeo 0.5 created a database in an empty directory).
- **A property set to `None` is not stored**: `upsert_nodes()` with `{"a": None}` leaves `a` out, and updating a property to `None` removes it. Properties grafeo 0.5 stored as null are gone after the migration.
- **Properties can be exact decimals**: a `decimal.Decimal` property is read back as a `Decimal` with its scale, and in GQL a number with a point (`1.50`) is a decimal, so GQL `structured_query()` results and properties written by GQL can hold `Decimal` values. Cypher `1.5` stays a float, and a filter with a float or a decimal matches a stored decimal of the same value.
- **GQL `ORDER BY` puts nulls last in both directions**: Cypher keeps nulls first when descending. Write `ORDER BY x DESC NULLS FIRST` for the descending order of grafeo 0.5.
- **Overflow and division by zero fail the query**: integer and decimal overflow and division by zero raise `grafeo.GrafeoError`; they gave null.
- **GQL refuses a filter right after a write**: a `WHERE` or `FILTER` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is a syntax error.
- **`get_schema()` label entries list their properties**: on grafeo 0.6.2 each label entry carries a `properties` list, and labels without nodes are left out.

### Fixed

- **`vector_query()` returns entities only**: it also returned text chunks, which LlamaIndex embeds, so chunks took the seed slots of `VectorContextRetriever` and `GrafeoPGRetriever` and found no relations. Chunks are left out, as in LlamaIndex's Neo4j store; their embeddings are still stored.
- **`vector_query()` applies `filters` and `node_ids`**: both were ignored, so `index.as_retriever(filters=...)` returned unfiltered results. They narrow the search inside the HNSW index (AND of `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`), and other filters raise `NotImplementedError`.
- **Dedup keeps the identity of the existing node**: a merge overwrote its ID and name with the new node's, so after a reopen the original ID found nothing and relations naming it were skipped. The existing node now keeps both, and the merged ID is saved as an alias (`li_aliases`) that resolves in `get()`, `get_triplets()`, `delete()` and relations, also after a reopen.
- **A `ChunkNode` with its own label stays a chunk**: a chunk labelled other than `text_chunk` was read back as an `EntityNode`, found by `vector_query()`, and could take entities in by dedup. Chunks are now marked by `li_kind`, whatever their label.
- **A node upserted with a new label moves to it**: the LlamaIndex label changed but the Grafeo label stayed, so the schema and `MATCH (n:NEW)` still showed the old one.
- **`StorageContext.from_defaults(persist_dir=...)` loads after `persist()`**: the database was written to `property_graph_store.json`, which LlamaIndex then read as JSON and failed with `UnicodeDecodeError`. A `.json` path is now saved with `.grafeo` (`property_graph_store.grafeo`).
- **Property keys with spaces or hyphens work in filters**: `get(properties=...)` and `delete(properties=...)` raised `ValueError` for keys such as `file name`, which upserts stored. Any key without a backtick is accepted.
- **`db_path` accepts a `pathlib.Path`**: it raised `TypeError`; `persist()` already took one.
- **`structured_query()` takes a query wrapped in a code fence**: LLMs often answer `TextToCypherRetriever` with the query inside a markdown fence (```` ```cypher ... ``` ````), which failed with `GRAFEO-Q001: syntax error`. One fence around the whole query, with or without a language tag, is removed before it runs; text outside a fence is not.
- **Building an index no longer warns about `SOURCE` relations**: `PropertyGraphIndex.from_documents()` emitted `UserWarning: Skipping relation 'SOURCE'` once per document, for the chunk-to-document relation that LlamaIndex's default `ImplicitPathExtractor` adds; documents are not graph nodes, so it is skipped quietly. A relation whose source or target is missing otherwise still warns.
- **`examples/graphrag_migration.py` reads current GraphRAG output**: it looked only for the `create_final_*.parquet` files of GraphRAG before 1.0, so it loaded nothing from a GraphRAG 1.0 or later run, and it linked no community to its members. It now reads both layouts, links members by title, and adds each community report's title, summary and rank (tested on GraphRAG 3.3 output).

### Dependencies

- **grafeo 0.6** (requires `>=0.6.2,<0.7`, tested on 0.6.2 and 0.6.3; 0.2.0 required `>=0.5`): grafeo 0.6 migrates `.grafeo` files and 0.5 WAL directories to its new format on the first read-write open. The old database is kept as `<path>.pre-0.6`, which grafeo 0.5 can still open.
- **llama-index-core 0.14** (requires `>=0.14,<1`, unchanged): locked dependencies are at their latest compatible versions (llama-index-workflows 2.26, pandas 3.1, pydantic 2.14).
- **Development tools**: grafeo-datasets 0.6.3, llama-index-llms-mistralai 0.12 and llama-index-embeddings-mistralai 0.7 (for the examples), pytest 9.2, ruff 0.17, `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.0`, `codecov/codecov-action@v7`, `actions/upload-artifact@v7` and `actions/download-artifact@v8`.

## [0.2.1] - Unreleased (ships in 0.3.0)

Fix release, never published on its own: built against grafeo 0.5.44, it fixes `delete()`, which failed on grafeo 0.5.44 for any node with relations, plus query, cache and threading bugs found while adding tests for them. Its changes ship in 0.3.0 on grafeo 0.6; the last release for grafeo 0.5 is 0.2.0.

### Added

- **Engine contract tests**: `tests/test_grafeo_contract.py` checks every grafeo call, query shape, result format and error the package relies on, with failure messages that name the affected store method.
- **Deletes, queries and reopened stores are tested**: deletes with relations, writes outside the store, relation labels, query parameters, embedding dimensions, file-backed stores closed and reopened, a database written by grafeo 0.5.44, concurrent writes and reads, and an end-to-end `PropertyGraphIndex` build, retrieve and delete with mock models.
- **`query_language` store option**: `"gql"` (the default in this version) or `"cypher"`, plus a per-call `language` argument on `structured_query()` and `astructured_query()`, so Cypher from `TextToCypherRetriever` and `CypherTemplateRetriever` runs as Cypher. A query starting with `g.` still runs as Gremlin, and an unknown language raises `ValueError`.
- **Scheduled run against the newest grafeo**: the `grafeo-latest.yml` workflow runs the suite weekly and on demand against the newest grafeo on PyPI, pre-releases included, both within the supported range and outside it.

### Changed

- **Query values are query parameters**: every query passes values as Grafeo query parameters instead of escaped literals, which also closes the escaping gaps for newlines and control characters.
- **`get_triplets(relation_names=...)` matches normalized labels**: a label matches as given or as normalized on upsert (`"works at"` finds `WORKS_AT`), like `delete(relation_names=...)`.
- **`GrafeoPGRetriever` reads only the nodes it ranks**: it read every node of the graph on every query to map PageRank scores. It now looks up only the nodes of the retrieved triplets, in PageRank scores the store caches until the graph changes.
- **README and examples use `.grafeo` paths**: other paths create a WAL directory, deprecated from grafeo 0.6. The README documents the grafeo 0.6 file migration.
- **Dev dependencies use `[dependency-groups]`**: they moved from `[tool.uv] dev-dependencies`, which uv deprecates.

### Fixed

- **`delete()` removes nodes that have relations**: by `ids`, `entity_names` or `properties` it raised `GrafeoError: Cannot delete node with N connected edge(s)` on grafeo 0.5.44; it now deletes the matching nodes and every relation attached to them in one `DETACH DELETE`. It removes every node with a matching name or ID, not only the first, and evicts every cached ID of a deleted node.
- **Relation labels with punctuation work**: labels with characters such as `/`, `(`, `'` or `.` (common in LLM-extracted labels) made `upsert_relations()`, `get_triplets(relation_names=...)` and `delete(relation_names=...)` fail with a syntax error.
- **`structured_query()` no longer rewrites the query text**: `param_map` was substituted into the query, so `$name` also rewrote `$name2` and a `$name` inside a string literal. Parameters are now passed to Grafeo as query parameters, for Gremlin too.
- **`None` values and keyword keys work in filters**: `get()`, `get_triplets()` and `delete()` failed for a property value of `None` and for property keys that are query keywords (`group`).
- **`get_triplets()` applies `limit` once, after filtering**: it applied `limit` before the `properties` filter, so it could miss matches, and once per relation name, so it could return more than `limit` triplets.
- **Upserts work after deletes made outside the store**: after nodes were deleted through `structured_query()` or `client`, upserts raised `GrafeoError: node N does not exist` from stale cached IDs. Cached IDs are now checked before use.
- **Relations to Grafeo node 0 are kept**: `upsert_relations()` skipped a relation whose source or target was Grafeo node 0.
- **Concurrent upserts create a node once**: `upsert_nodes()` calls for the same node from several threads created duplicates. Writes are now serialized by a lock.
- **`vector_query()` is safe during concurrent upserts**: it could fail with `Set changed size during iteration` while another thread upserted a node with a new label.
- **Embeddings of the wrong size are refused**: `upsert_nodes()` raises `ValueError` before writing anything when an embedding's length differs from `embedding_dimensions`. Such an embedding was stored and left `vector_query()` empty for its label without an error, or crashed Grafeo's vector search with a `PanicException` when dedup was enabled.
- **A mismatched `embedding_dimensions` on reopen is reported**: reopening a database with dimensions that do not match its stored embeddings left `vector_query()` empty without explanation. It now emits a `UserWarning` naming the label.
- **`get_rel_map()` follows relations in both directions**: it followed only outgoing relations, so a node that is only the target of its relations (a company people work at) got no context. Triplets keep their stored direction and each relation is returned once, like LlamaIndex's reference stores.

### Dependencies

- **grafeo 0.5.44** (requires `>=0.5.44,<0.6`, was `>=0.5`): the upper bound keeps this release off grafeo 0.6 and its one-way file migration.

## [0.2.0] - 2026-04-12

Feature release: aligned with grafeo 0.5.37, with new features, test coverage and documentation improvements.

### Added

- Auto-detect embedding dimensions from model probe at init: listed here in error, it did not ship in 0.2.0 and ships in 0.3.0 as `embed_model`
- Entity deduplication: configurable cosine similarity threshold merges near-duplicate entities
- Timestamps: `created_at` and `updated_at` on all nodes, UTC ISO 8601
- Convenience API: `node_count`, `edge_count`, `summary()`, `neighbors()`, `persist()`, `close()`
- `GrafeoPGRetriever`: custom retriever with PageRank reranking and multi-hop expansion
- Context manager support for automatic cleanup
- Persistence round-trip tests: nodes, edges and vector search survive close/reopen
- Async retrieve tests: `aretrieve_from_graph()` verified to match sync results and scores
- Performance tests: 1000 nodes + 2000 relations vector query under 2s, concurrent upsert with 2 threads
- Embedding dimension mismatch test: `ValueError` on wrong dimensions in `vector_query`
- Missing relation endpoint tests: `UserWarning` emitted when source/target nodes missing
- Special character round-trip tests: quotes, backslashes, unicode verified through upsert/get
- Schema completeness test: `get_schema_str()` includes all labels and edge types
- Empty graph retriever test: returns empty list, no crash
- Dedup threshold boundary test: controlled embeddings verify merge/split consistency
- `examples/mock_embedding_demo.py`: fully runnable demo with no API key required

### Changed

- Requires grafeo >=0.5 (tested against 0.5.37)
- README: added Persistence, Deduplication, and Relation Upsert Behavior sections
- README: added `dedup_threshold` to constructor signature in API Reference
- README: mock embedding demo listed first in examples
- 124 tests passing, 96% coverage

### Fixed

- `upsert_relations()` now emits `UserWarning` when source or target node is missing (was silent skip)
- `vector_query()` now validates embedding dimensions, raises `ValueError` on mismatch

## [0.1.0] - 2026-02-10

- Initial release with `GrafeoPropertyGraphStore` implementing all 8 LlamaIndex PropertyGraphStore methods.

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
