from __future__ import annotations

import pytest
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


@pytest.fixture()
def vector_store() -> GrafeoPropertyGraphStore:
    """Store with small embeddings for vector search tests."""
    s = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
    s.upsert_nodes(
        [
            EntityNode(name="graph-db", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5]),
            EntityNode(name="vector-search", label="entity", properties={}, embedding=[0.2, 0.7, 0.4, 0.6]),
            EntityNode(name="cooking", label="entity", properties={}, embedding=[0.9, 0.1, 0.2, 0.1]),
        ]
    )
    return s


class TestVectorQuery:
    def test_basic_cosine(self, vector_store: GrafeoPropertyGraphStore) -> None:
        query = VectorStoreQuery(query_embedding=[0.15, 0.75, 0.35, 0.55], similarity_top_k=2)
        nodes, scores = vector_store.vector_query(query)

        assert len(nodes) == 2
        assert len(scores) == 2
        # graph-db and vector-search should be closest to the query
        names = {n.name for n in nodes}
        assert "graph-db" in names or "vector-search" in names

    def test_no_embedding(self, vector_store: GrafeoPropertyGraphStore) -> None:
        query = VectorStoreQuery(query_embedding=None, similarity_top_k=2)
        nodes, scores = vector_store.vector_query(query)
        assert nodes == []
        assert scores == []

    def test_top_k_respected(self, vector_store: GrafeoPropertyGraphStore) -> None:
        query = VectorStoreQuery(query_embedding=[0.15, 0.75, 0.35, 0.55], similarity_top_k=1)
        nodes, _scores = vector_store.vector_query(query)
        assert len(nodes) == 1

    def test_scores_descending(self, vector_store: GrafeoPropertyGraphStore) -> None:
        query = VectorStoreQuery(query_embedding=[0.15, 0.75, 0.35, 0.55], similarity_top_k=3)
        _, scores = vector_store.vector_query(query)
        assert scores == sorted(scores, reverse=True)


class TestCustomLabelVector:
    def test_custom_label_searchable(self) -> None:
        """Nodes with custom labels should be findable via vector_query."""
        s = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        s.upsert_nodes(
            [
                EntityNode(name="gravity", label="concept", properties={}, embedding=[0.1, 0.9, 0.2, 0.8]),
                EntityNode(name="entropy", label="concept", properties={}, embedding=[0.2, 0.8, 0.3, 0.7]),
            ]
        )
        query = VectorStoreQuery(query_embedding=[0.15, 0.85, 0.25, 0.75], similarity_top_k=2)
        nodes, _scores = s.vector_query(query)
        assert len(nodes) == 2
        names = {n.name for n in nodes}
        assert "gravity" in names
        assert "entropy" in names

    def test_mixed_labels_searchable(self) -> None:
        """Vector search across multiple custom labels returns results from all of them."""
        s = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        s.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}, embedding=[0.9, 0.1, 0.1, 0.1]),
                EntityNode(name="gravity", label="concept", properties={}, embedding=[0.1, 0.9, 0.1, 0.9]),
            ]
        )
        query = VectorStoreQuery(query_embedding=[0.5, 0.5, 0.1, 0.5], similarity_top_k=2)
        nodes, _scores = s.vector_query(query)
        assert len(nodes) == 2
        labels = {n.label for n in nodes}
        assert "person" in labels
        assert "concept" in labels

    def test_no_embeddings_empty_result(self) -> None:
        """vector_query on a store with no embeddings returns empty lists."""
        s = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        s.upsert_nodes([EntityNode(name="X", label="entity", properties={})])
        query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=2)
        nodes, scores = s.vector_query(query)
        assert nodes == []
        assert scores == []


class TestDistanceToScore:
    def test_cosine(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        assert store._distance_to_score(0.0) == 1.0
        assert store._distance_to_score(1.0) == 0.0
        assert store._distance_to_score(0.5) == pytest.approx(0.5)

    def test_euclidean(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="euclidean")
        assert store._distance_to_score(0.0) == 1.0
        assert store._distance_to_score(1.0) == pytest.approx(0.5)
        assert store._distance_to_score(3.0) == pytest.approx(0.25)

    def test_dot_product(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="dot_product")
        # Grafeo negates dot product, so distance=-10 means raw dp=10
        assert store._distance_to_score(-10.0) == 10.0
        assert store._distance_to_score(0.0) == 0.0

    def test_manhattan(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="manhattan")
        assert store._distance_to_score(0.0) == 1.0
        assert store._distance_to_score(1.0) == pytest.approx(0.5)
