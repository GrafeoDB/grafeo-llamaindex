from __future__ import annotations

from collections import Counter
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
    """Create a mock embed model that returns a fixed query embedding, and one for each relation label."""
    mock = MagicMock()
    mock.get_agg_embedding_from_queries.return_value = [0.85, 0.15, 0.25, 0.1]
    mock.get_text_embedding_batch.side_effect = lambda texts: [[0.5, 0.5, 0.5, 0.5] for _ in texts]
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


class TestRetrieverEdgeCases:
    """T10: Retriever on empty graph should return empty list, not crash."""

    def test_empty_graph_returns_empty(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        retriever = GrafeoPGRetriever(
            graph_store=store,
            embed_model=_mock_embed_model(),
            similarity_top_k=2,
            include_text=False,
        )
        results = retriever.retrieve("anything at all")
        assert results == []


class TestDedupThresholdBoundary:
    """T11: Dedup threshold boundary with controlled embeddings.

    Insert two entities whose cosine similarity equals the dedup_threshold.
    Verify consistent merge/split behavior.
    """

    def test_at_threshold_merges(self) -> None:
        """Entities at or above threshold should be merged."""
        # Cosine distance 0.0 for identical vectors, so similarity = 1.0
        store = GrafeoPropertyGraphStore(
            embedding_dimensions=DIMS,
            embedding_metric="cosine",
            dedup_threshold=0.99,
        )
        embedding = [1.0, 0.0, 0.0, 0.0]
        store.upsert_nodes([EntityNode(name="A", label="entity", properties={}, embedding=embedding)])
        store.upsert_nodes([EntityNode(name="B", label="entity", properties={}, embedding=embedding)])
        # Identical embeddings exceed any threshold below 1.0
        assert store.node_count == 1

    def test_just_below_threshold_keeps_separate(self) -> None:
        """Entities below threshold should remain separate."""
        # Use very high threshold so even fairly similar vectors stay separate
        store = GrafeoPropertyGraphStore(
            embedding_dimensions=DIMS,
            embedding_metric="cosine",
            dedup_threshold=0.9999,
        )
        store.upsert_nodes([EntityNode(name="A", label="entity", properties={}, embedding=[1.0, 0.0, 0.0, 0.0])])
        # This vector has cosine similarity ~0.707 with [1,0,0,0]
        store.upsert_nodes([EntityNode(name="B", label="entity", properties={}, embedding=[0.707, 0.707, 0.0, 0.0])])
        assert store.node_count == 2


def _old_pagerank_scores(store: GrafeoPropertyGraphStore) -> dict[str, float]:
    """The 0.2.0 implementation: PageRank, then read every node to map it to its LlamaIndex ID."""
    db = store.client
    raw = db.algorithms.pagerank()
    if not raw:
        return {}
    max_score = max(raw.values()) or 1.0
    result: dict[str, float] = {}
    for gid, score in raw.items():
        node = db.get_node(gid)
        if node is None:
            continue
        result[node.properties().get("li_id", str(gid))] = score / max_score
    return result


class _OldScoringRetriever(GrafeoPGRetriever):
    def _pagerank_scores(self, node_ids: set[str]) -> dict[str, float]:
        return _old_pagerank_scores(self._grafeo_store)


class _CountingDB:
    """Delegates to a GrafeoDB and counts the attributes the store reads."""

    def __init__(self, db: object) -> None:
        self._db = db
        self.calls: Counter[str] = Counter()

    def __getattr__(self, name: str) -> object:
        self.calls[name] += 1
        return getattr(self._db, name)


def _hub_store() -> GrafeoPropertyGraphStore:
    """A star around Hub plus a chain, so PageRank differs between nodes, and 300 unconnected nodes."""
    store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
    names = ["Hub", "A", "B", "C", "D", "E"]
    store.upsert_nodes(
        [EntityNode(name=n, embedding=[0.9 - 0.1 * i, 0.1 + 0.1 * i, 0.2, 0.1]) for i, n in enumerate(names)]
    )
    store.upsert_nodes([EntityNode(name=f"x{i}") for i in range(300)])
    store.upsert_relations(
        [Relation(label="LINKS", source_id=n, target_id="Hub") for n in ["A", "B", "C"]]
        + [Relation(label="NEXT", source_id="Hub", target_id="D"), Relation(label="NEXT", source_id="D", target_id="E")]
    )
    # A node written outside the store, without li_id: named by its Grafeo ID in triplets.
    outsider = store.client.create_node(labels=["entity"], properties={})
    store.client.create_edge(
        source_id=outsider.id, target_id=store._find_node_id("E"), edge_type="POINTS", properties={}
    )
    return store


def _gid(store: GrafeoPropertyGraphStore, li_id: str) -> int:
    gid = store._find_node_id(li_id)
    assert gid is not None
    return gid


class TestPageRankScores:
    def test_same_scores_as_before(self) -> None:
        store = _hub_store()
        retriever = GrafeoPGRetriever(graph_store=store, embed_model=_mock_embed_model())
        old = _old_pagerank_scores(store)
        triplet_ids = {t[0].id for t in store.get_triplets()} | {t[2].id for t in store.get_triplets()}
        new = retriever._pagerank_scores(triplet_ids)
        assert new.keys() == triplet_ids
        assert new == pytest.approx({li_id: old.get(li_id, 0.0) for li_id in triplet_ids})

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.7])
    def test_same_ranking_as_before(self, alpha: float) -> None:
        store = _hub_store()
        results = [
            cls(
                graph_store=store,
                embed_model=_mock_embed_model(),
                similarity_top_k=3,
                path_depth=2,
                alpha=alpha,
                include_text=False,
            ).retrieve("q")
            for cls in (GrafeoPGRetriever, _OldScoringRetriever)
        ]
        new, old = results
        assert [r.text for r in new] == [r.text for r in old]
        assert [r.score for r in new] == pytest.approx([r.score for r in old])

    def test_second_query_reads_no_node_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _hub_store()
        counting = _CountingDB(store.client)
        monkeypatch.setattr(store, "_db", counting)
        retriever = GrafeoPGRetriever(
            graph_store=store, embed_model=_mock_embed_model(), similarity_top_k=3, path_depth=2, include_text=False
        )

        assert retriever.retrieve("q")
        assert counting.calls["algorithms"] == 1
        first_reads = counting.calls["get_node"]
        assert first_reads < 100  # the 0.2.0 retriever read all 307 nodes on every query

        assert retriever.retrieve("q")
        assert counting.calls["algorithms"] == 1  # PageRank came from the cache
        assert counting.calls["get_node"] - first_reads < 100

    def test_writes_through_the_store_invalidate(self) -> None:
        store = _hub_store()
        before = store._pagerank()
        assert store._pagerank() is before  # cached

        store.upsert_relations([Relation(label="LINKS", source_id="E", target_id="A")])
        assert store._pagerank_cache is None
        after_upsert = store._pagerank()
        assert after_upsert[_gid(store, "A")] != before[_gid(store, "A")]

        store.upsert_nodes([EntityNode(name="new")])
        assert store._pagerank_cache is None
        assert _gid(store, "new") in store._pagerank()

        store.delete(ids=["Hub"])
        assert store._pagerank_cache is None
        assert store._find_node_id("Hub") is None
        assert len(store._pagerank()) == store.node_count

        store._pagerank()
        store.structured_query("MATCH (n) WHERE n.name = 'D' DETACH DELETE n")
        assert store._pagerank_cache is None
        assert len(store._pagerank()) == store.node_count

    def test_writes_through_client_are_detected(self) -> None:
        store = _hub_store()
        before = store._pagerank()
        a, e = _gid(store, "A"), _gid(store, "E")
        store.client.create_edge(source_id=e, target_id=a, edge_type="LINKS", properties={})  # bypasses the store
        after = store._pagerank()
        assert after is not before
        assert after[a] > before[a]

    def test_reads_keep_the_cache(self) -> None:
        store = _hub_store()
        before = store._pagerank()
        store.get(ids=["Hub"])
        store.get_triplets(entity_names=["Hub"])
        store.get_rel_map(store.get(ids=["Hub"]), depth=2)
        assert store._pagerank() is before

    def test_empty_graph(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        assert store._pagerank() == {}
        retriever = GrafeoPGRetriever(graph_store=store, embed_model=_mock_embed_model())
        assert retriever._pagerank_scores({"anything"}) == {}


class TestRelevanceRanking:
    """The triplet that answers ranks above a hub's triplets, with PageRank on."""

    @staticmethod
    def _store() -> GrafeoPropertyGraphStore:
        # Gilfoyle -[BUILT]-> Anton answers the query; Palo Alto is a hub many people live in.
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name="Gilfoyle", embedding=[0.9, 0.1, 0.0, 0.0]),
                EntityNode(name="Anton", embedding=[0.95, 0.0, 0.05, 0.0]),
                EntityNode(name="Palo Alto", embedding=[0.0, 0.0, 1.0, 0.0]),
            ]
            + [EntityNode(name=f"resident{i}", embedding=[0.0, 0.0, 0.0, 1.0]) for i in range(6)]
        )
        store.upsert_relations(
            [
                Relation(label="BUILT", source_id="Gilfoyle", target_id="Anton"),
                Relation(label="LIVES_IN", source_id="Gilfoyle", target_id="Palo Alto"),
            ]
            + [Relation(label="LIVES_IN", source_id=f"resident{i}", target_id="Palo Alto") for i in range(6)]
        )
        return store

    @staticmethod
    def _embed_model() -> MagicMock:
        label_vectors = {"built": [1.0, 0.0, 0.0, 0.0], "lives in": [0.0, 0.0, 0.0, 1.0]}
        mock = MagicMock()
        mock.get_agg_embedding_from_queries.return_value = [1.0, 0.0, 0.0, 0.0]  # "what did Gilfoyle build?"
        mock.get_text_embedding_batch.side_effect = lambda texts: [label_vectors[t] for t in texts]
        return mock

    def test_answer_outranks_hub(self) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=self._store(), embed_model=self._embed_model(), similarity_top_k=2, include_text=False
        )
        results = retriever.retrieve("What did Gilfoyle build?")
        assert results[0].text == "Gilfoyle -> BUILT -> Anton"

    def test_labels_are_embedded_once(self) -> None:
        embed_model = self._embed_model()
        retriever = GrafeoPGRetriever(
            graph_store=self._store(), embed_model=embed_model, similarity_top_k=2, include_text=False
        )
        retriever.retrieve("q1")
        retriever.retrieve("q2")
        embedded = [text for call in embed_model.get_text_embedding_batch.call_args_list for text in call.args[0]]
        assert sorted(embedded) == ["built", "lives in"]

    def test_limit_applies_after_ranking(self) -> None:
        retriever = GrafeoPGRetriever(
            graph_store=self._store(), embed_model=self._embed_model(), similarity_top_k=2, limit=1, include_text=False
        )
        assert [r.text for r in retriever.retrieve("q")] == ["Gilfoyle -> BUILT -> Anton"]
