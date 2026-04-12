from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import QueryBundle

from grafeo_llamaindex import GrafeoPGRetriever, GrafeoPropertyGraphStore

DIMS = 4


@pytest.fixture()
def async_store() -> GrafeoPropertyGraphStore:
    """Store with embeddings for async retriever tests."""
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
    mock = MagicMock()
    mock.get_agg_embedding_from_queries.return_value = [0.85, 0.15, 0.25, 0.1]
    return mock


class TestAsyncRetrieve:
    """T2: Async retrieve should produce the same results as sync retrieve."""

    @pytest.mark.asyncio
    async def test_aretrieve_returns_results(self, async_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=async_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        qb = QueryBundle(query_str="people query")
        results = await retriever.aretrieve_from_graph(qb)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_aretrieve_matches_sync(self, async_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=async_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        qb_sync = QueryBundle(query_str="people query")
        qb_async = QueryBundle(query_str="people query")

        sync_results = retriever.retrieve_from_graph(qb_sync)
        async_results = await retriever.aretrieve_from_graph(qb_async)

        assert len(sync_results) == len(async_results)

        sync_texts = sorted(r.text for r in sync_results)
        async_texts = sorted(r.text for r in async_results)
        assert sync_texts == async_texts

    @pytest.mark.asyncio
    async def test_aretrieve_scores_match_sync(self, async_store: GrafeoPropertyGraphStore) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=async_store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            path_depth=1,
            include_text=False,
        )
        qb_sync = QueryBundle(query_str="people query")
        qb_async = QueryBundle(query_str="people query")

        sync_results = retriever.retrieve_from_graph(qb_sync)
        async_results = await retriever.aretrieve_from_graph(qb_async)

        sync_scores = sorted(r.score for r in sync_results if r.score is not None)
        async_scores = sorted(r.score for r in async_results if r.score is not None)
        assert sync_scores == pytest.approx(async_scores)
