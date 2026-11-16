"""Embedding dimensions read from an embedding model (``embed_model``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore
from grafeo_llamaindex.property_graph_store import DEFAULT_EMBEDDING_DIMENSIONS


class _CountingEmbedding(MockEmbedding):
    """MockEmbedding that counts how often it embeds a text."""

    calls: int = 0

    def _get_text_embedding(self, text: str) -> list[float]:
        self.calls += 1
        return super()._get_text_embedding(text)


class _EmptyEmbedding(MockEmbedding):
    def _get_text_embedding(self, text: str) -> list[float]:
        return []


class TestEmbedModel:
    def test_init_reads_dimensions_from_model(self) -> None:
        store = GrafeoPropertyGraphStore(embed_model=MockEmbedding(embed_dim=8))
        assert store.embedding_dimensions == 8

    def test_init_probes_model_once(self) -> None:
        model = _CountingEmbedding(embed_dim=8)
        GrafeoPropertyGraphStore(embed_model=model)
        assert model.calls == 1

    def test_init_without_model_uses_default(self) -> None:
        assert GrafeoPropertyGraphStore().embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS == 1536

    def test_init_explicit_dimensions_without_model(self) -> None:
        assert GrafeoPropertyGraphStore(embedding_dimensions=4).embedding_dimensions == 4

    def test_init_matching_dimensions_and_model(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=8, embed_model=MockEmbedding(embed_dim=8))
        assert store.embedding_dimensions == 8

    def test_init_conflicting_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match=r"embedding_dimensions=4 .* 8 dimensions"):
            GrafeoPropertyGraphStore(embedding_dimensions=4, embed_model=MockEmbedding(embed_dim=8))

    def test_init_empty_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="empty embedding"):
            GrafeoPropertyGraphStore(embed_model=_EmptyEmbedding(embed_dim=8))

    def test_init_conflict_leaves_no_database(self, tmp_path: Path) -> None:
        path = tmp_path / "kg.grafeo"
        with pytest.raises(ValueError):
            GrafeoPropertyGraphStore(db_path=str(path), embedding_dimensions=4, embed_model=MockEmbedding(embed_dim=8))
        assert not path.exists()

    def test_upsert_and_query_with_model_embeddings(self) -> None:
        model = MockEmbedding(embed_dim=8)
        store = GrafeoPropertyGraphStore(embed_model=model)
        embedding: list[Any] = model.get_text_embedding("Alice")
        store.upsert_nodes([EntityNode(name="Alice", label="person", embedding=embedding)])
        nodes, _scores = store.vector_query(VectorStoreQuery(query_embedding=embedding, similarity_top_k=1))
        assert [n.id for n in nodes] == ["Alice"]

    def test_reopen_with_model(self, tmp_path: Path) -> None:
        path = str(tmp_path / "kg.grafeo")
        model = MockEmbedding(embed_dim=8)
        embedding = model.get_text_embedding("Alice")
        with GrafeoPropertyGraphStore(db_path=path, embed_model=model) as store:
            store.upsert_nodes([EntityNode(name="Alice", label="person", embedding=embedding)])
        with GrafeoPropertyGraphStore(db_path=path, embed_model=model) as store:
            nodes, _scores = store.vector_query(VectorStoreQuery(query_embedding=embedding, similarity_top_k=1))
            assert [n.id for n in nodes] == ["Alice"]
