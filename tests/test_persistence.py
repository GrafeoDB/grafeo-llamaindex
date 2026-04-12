from __future__ import annotations

from pathlib import Path

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


class TestPersistenceRoundTrip:
    """T7: Create, populate, close, reopen, verify data and indexes survive."""

    def test_nodes_survive_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist.db")

        nodes = [EntityNode(name=f"entity_{i}", label="entity", properties={"idx": i}) for i in range(20)]
        relations = [
            Relation(
                label="LINKS_TO",
                source_id=f"entity_{i}",
                target_id=f"entity_{i + 1}",
                properties={},
            )
            for i in range(10)
        ]

        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(nodes)
        store.upsert_relations(relations)

        original_node_count = store.node_count
        original_edge_count = store.edge_count
        assert original_node_count == 20
        assert original_edge_count == 10

        store.close()

        # Reopen
        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        assert store2.node_count == original_node_count
        assert store2.edge_count == original_edge_count
        store2.close()

    def test_vector_search_survives_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist_vec.db")

        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name=f"node_{i}", label="entity", properties={}, embedding=[float(i) / 20] * DIMS)
                for i in range(20)
            ]
        )
        store.upsert_relations(
            [
                Relation(
                    label="LINKS_TO",
                    source_id=f"node_{i}",
                    target_id=f"node_{i + 1}",
                    properties={},
                )
                for i in range(10)
            ]
        )
        store.close()

        # Reopen: vector indexes are rediscovered automatically, no probe needed
        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)

        query = VectorStoreQuery(query_embedding=[0.5] * DIMS, similarity_top_k=5)
        nodes, scores = store2.vector_query(query)
        assert len(nodes) > 0
        assert len(scores) == len(nodes)
        store2.close()

    def test_vector_index_labels_rediscovered(self, tmp_path: Path) -> None:
        """_vector_indexed_labels is populated from existing indexes on reopen."""
        db_path = str(tmp_path / "rediscover.db")
        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name="a", label="person", properties={}, embedding=[1.0, 0.0, 0.0, 0.0]),
                EntityNode(name="b", label="company", properties={}, embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        assert "person" in store._vector_indexed_labels
        assert "company" in store._vector_indexed_labels
        store.close()

        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        assert "person" in store2._vector_indexed_labels
        assert "company" in store2._vector_indexed_labels
        store2.close()

    def test_persist_in_memory_then_reopen(self, tmp_path: Path) -> None:
        """In-memory store can be persisted and then reopened."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        store.upsert_nodes([EntityNode(name=f"e_{i}", label="entity", properties={"idx": i}) for i in range(20)])
        store.upsert_relations(
            [
                Relation(
                    label="REL",
                    source_id=f"e_{i}",
                    target_id=f"e_{i + 1}",
                    properties={},
                )
                for i in range(10)
            ]
        )

        persist_path = str(tmp_path / "saved.db")
        store.persist(persist_path)
        store.close()

        store2 = GrafeoPropertyGraphStore(db_path=persist_path, embedding_dimensions=DIMS)
        assert store2.node_count == 20
        assert store2.edge_count == 10
        store2.close()
