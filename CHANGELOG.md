# Changelog

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
