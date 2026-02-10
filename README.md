# grafeo-llamaindex

[LlamaIndex](https://github.com/run-llama/llama_index) integration for the [Grafeo](https://github.com/GrafeoDB/grafeo) graph database.

Implements `GrafeoPropertyGraphStore`, a LlamaIndex `PropertyGraphStore` backend that supports both structured and vector queries.

## Status

Work in progress.

## Features (planned)

- Full `PropertyGraphStore` implementation (all 8 abstract methods)
- Structured graph queries via GQL
- Vector similarity queries via built-in HNSW indexes
- EntityNode, ChunkNode, and Relation mapping
- Lazy vector index creation, eager property indexes
- Zero infrastructure &mdash; Grafeo is embedded, no external services required

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

## License

Apache-2.0 &mdash; see [LICENSE](LICENSE) for details.
