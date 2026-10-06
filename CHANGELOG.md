# Changelog

## 0.2.1 - Unreleased

Works with grafeo 0.5.44. Fixes `delete()`, which failed on grafeo 0.5.44 for any node with relations, plus query, cache and threading bugs found while adding tests for them.

### Added

- Engine contract tests (`tests/test_grafeo_contract.py`): every grafeo call, query shape, result format and error the package relies on, with failure messages that name the affected store method
- Tests for deletes with relations, writes outside the store, relation labels, query parameters, embedding dimensions, file-backed stores closed and reopened (single `.grafeo` file and WAL directory), a database written by grafeo 0.5.44, concurrent writes and reads, and an end-to-end `PropertyGraphIndex` build, retrieve and delete with mock models
- `query_language` store option (`"gql"` or `"cypher"`) and a per-call `language` argument on `structured_query()` and `astructured_query()`, so Cypher from LlamaIndex's `TextToCypherRetriever` and `CypherTemplateRetriever` can run as Cypher. The default stays `"gql"` (`DEFAULT_QUERY_LANGUAGE`); a query starting with `g.` still runs as Gremlin. An unknown language raises `ValueError`
- Scheduled CI workflow (`grafeo-latest.yml`, weekly and on demand) that runs the suite against the newest grafeo on PyPI, pre-releases included, both within the supported range and outside it

### Fixed

- `delete()` by `ids`, `entity_names` or `properties` raised `GrafeoError: Cannot delete node with N connected edge(s)` on grafeo 0.5.44, which no longer deletes a node that still has edges. It now deletes the matching nodes and every relation attached to them in one `DETACH DELETE`, removes all nodes with a matching name or ID (not only the first), and evicts every cached ID of a deleted node
- Relation labels with characters such as `/`, `(`, `'` or `.` (common in LLM-extracted labels) made `upsert_relations()`, `get_triplets(relation_names=...)` and `delete(relation_names=...)` fail with a syntax error
- `structured_query()` substituted `param_map` into the query text: `$name` also rewrote `$name2`, and a `$name` inside a string literal was replaced. Parameters are now passed to Grafeo as query parameters, for Gremlin too
- `get()`, `get_triplets()` and `delete()` failed for a property value of `None` and for property keys that are query keywords (`group`). All values are now query parameters and keys are quoted
- `get_triplets()` applied `limit` before the `properties` filter, so it could miss matches, and applied it once per relation name, so it could return more than `limit` triplets
- After nodes were deleted through `structured_query()` or `client`, upserts raised `GrafeoError: node N does not exist` (grafeo 0.5.44) from stale cached IDs. Cached IDs are now checked before use
- `upsert_relations()` skipped a relation whose source or target was Grafeo node 0, when the lookup by name returned 0 and the fallback lookup by ID found nothing
- Concurrent `upsert_nodes()` calls for the same node from several threads created duplicates. Writes are now serialized by a lock
- `vector_query()` could fail with `Set changed size during iteration` while another thread upserted a node with a new label
- An embedding whose length differed from `embedding_dimensions` was stored, after which that label's vector index could not be built and `vector_query()` returned nothing for it, without an error. With dedup enabled it crashed Grafeo's vector search with a `PanicException`. `upsert_nodes()` now raises `ValueError` before writing anything
- Reopening a database with an `embedding_dimensions` that does not match its stored embeddings left `vector_query()` empty without explanation. It now emits a `UserWarning` naming the label
- `get_rel_map()` followed only outgoing relations, so a node that is only the target of its relations (for example a company people work at) got no context. It now follows relations in both directions, like LlamaIndex's reference stores, still with one query per hop; triplets keep their stored direction and each relation is returned once

### Changed

- Requires grafeo `>=0.5.44,<0.6` (was `>=0.5`); tested against 0.5.44 and a grafeo 0.6 development build. The upper bound keeps this release off grafeo 0.6 and its one-way file migration; grafeo-llamaindex 0.3.0 moves to `grafeo~=0.6.0`
- All queries pass values as Grafeo query parameters instead of escaped literals (this also covers the escaping gaps for newlines and control characters)
- `get_triplets(relation_names=...)` matches a label as given or as normalized on upsert (`"works at"` finds `WORKS_AT`), like `delete(relation_names=...)`
- README and examples use `.grafeo` paths: other paths create a WAL directory, deprecated from grafeo 0.6. README documents the grafeo 0.6 file migration
- `GrafeoPGRetriever` read every node of the graph on every query to map PageRank scores to LlamaIndex IDs. It now looks up only the nodes of the retrieved triplets in PageRank scores the store caches, which writes through the store clear and direct writes through `client` invalidate by advancing the database epoch
- Dev dependencies moved from `[tool.uv] dev-dependencies` (deprecated by uv) to `[dependency-groups]`

## 0.2.0 - 2026-04-12

Aligned with grafeo 0.5.37. New features, test coverage and documentation improvements.

### Added

- Auto-detect embedding dimensions from model probe at init
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

### Fixed

- `upsert_relations()` now emits `UserWarning` when source or target node is missing (was silent skip)
- `vector_query()` now validates embedding dimensions, raises `ValueError` on mismatch

### Changed

- Requires grafeo >=0.5 (tested against 0.5.37)
- README: added Persistence, Deduplication, and Relation Upsert Behavior sections
- README: added `dedup_threshold` to constructor signature in API Reference
- README: mock embedding demo listed first in examples
- 124 tests passing, 96% coverage

## 0.1.0

- Initial release with `GrafeoPropertyGraphStore` implementing all 8 LlamaIndex PropertyGraphStore methods.
