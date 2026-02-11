from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import QueryBundle

from grafeo_llamaindex import GrafeoPGRetriever, GrafeoPropertyGraphStore

DIMS = 4


@pytest.fixture()
def retriever_store() -> GrafeoPropertyGraphStore:
    """Store with embeddings for retriever tests."""
    s = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
    s.upsert_nodes(
        [
            EntityNode(name="Alice", label="entity", properties={}, embedding=[0.9, 0.1, 0.2, 0.1]),
            EntityNode(name="Bob", label="entity", properties={}, embedding=[0.8, 0.2, 0.3, 0.1]),
            EntityNode(name="Acme", label="entity", properties={}, embedding=[0.1, 0.9, 0.1, 0.8]),
        ]
    )
    s.upsert_relations(
        [
            Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={}),
            Relation(label="WORKS_AT", source_id="Bob", target_id="Acme", properties={}),
        ]
    )
    return s


def _mock_embed_model() -> MagicMock:
    """Create a mock embed model that returns a fixed query embedding."""
    mock = MagicMock()
    mock.get_agg_embedding_from_queries.return_value = [0.85, 0.15, 0.25, 0.1]
    return mock


class TestGrafeoPGRetriever:
    def test_retrieve_returns_results(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        results = retriever.retrieve("people query")
        assert len(results) > 0

    def test_retrieve_returns_node_with_score(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        results = retriever.retrieve("people query")
        for r in results:
            assert r.score is not None
            assert r.score >= 0.0

    def test_scores_descending(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=3,
            path_depth=1,
            include_text=False,
        )
        results = retriever.retrieve("people query")
        scores = [r.score for r in results if r.score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_pagerank_disabled(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            use_pagerank=False,
            include_text=False,
        )
        results = retriever.retrieve("people query")
        assert len(results) > 0

    def test_empty_on_no_embeddings(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        store.upsert_nodes([EntityNode(name="X", label="entity", properties={})])
        retriever = GrafeoPGRetriever(
            graph_store=store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            include_text=False,
        )
        results = retriever.retrieve("anything")
        assert results == []

    def test_depth_expansion(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever_d1 = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        retriever_d2 = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=2,
            include_text=False,
        )
        results_d1 = retriever_d1.retrieve("people query")
        results_d2 = retriever_d2.retrieve("people query")
        # depth=2 should find at least as many triplets as depth=1
        assert len(results_d2) >= len(results_d1)

    def test_alpha_blending(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever_sim = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            alpha=1.0,  # pure similarity
            include_text=False,
        )
        retriever_pr = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            alpha=0.0,  # pure pagerank
            include_text=False,
        )
        results_sim = retriever_sim.retrieve("people query")
        results_pr = retriever_pr.retrieve("people query")
        # Both should return results, but scores may differ
        assert len(results_sim) > 0
        assert len(results_pr) > 0

    def test_retrieve_from_graph_directly(self, retriever_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=retriever_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        qb = QueryBundle(query_str="people query")
        results = retriever.retrieve_from_graph(qb)
        assert len(results) > 0
