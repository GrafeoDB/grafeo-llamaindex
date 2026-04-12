from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


@pytest.mark.slow
class TestLargeGraph:
    """T8: Insert 1000 EntityNode + 2000 Relation. Time vector_query."""

    def test_vector_query_performance(self) -> None:
        rng = random.Random(42)
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")

        nodes = [
            EntityNode(
                name=f"node_{i}",
                label="entity",
                properties={"idx": i},
                embedding=[rng.random() for _ in range(DIMS)],
            )
            for i in range(1000)
        ]
        store.upsert_nodes(nodes)

        relations = [
            Relation(
                label="LINKS_TO",
                source_id=f"node_{rng.randint(0, 999)}",
                target_id=f"node_{rng.randint(0, 999)}",
                properties={},
            )
            for _ in range(2000)
        ]
        store.upsert_relations(relations)

        assert store.node_count == 1000

        query_embedding = [rng.random() for _ in range(DIMS)]
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=10)

        start = time.perf_counter()
        result_nodes, scores = store.vector_query(query)
        elapsed = time.perf_counter() - start

        assert len(result_nodes) == 10
        assert len(scores) == 10
        assert elapsed < 2.0, f"vector_query took {elapsed:.2f}s, expected < 2s"


@pytest.mark.slow
class TestConcurrentUpsert:
    """T9: Two threads upserting different nodes. Verify no data loss."""

    def test_concurrent_upsert_no_data_loss(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)

        def upsert_batch(start: int, end: int) -> None:
            nodes = [EntityNode(name=f"node_{i}", label="entity", properties={"idx": i}) for i in range(start, end)]
            store.upsert_nodes(nodes)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(upsert_batch, 0, 500)
            future_b = pool.submit(upsert_batch, 500, 1000)
            future_a.result()
            future_b.result()

        assert store.node_count == 1000
