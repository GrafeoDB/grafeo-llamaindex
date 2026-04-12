from __future__ import annotations

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


@pytest.fixture()
def dedup_store() -> GrafeoPropertyGraphStore:
    """Store with dedup enabled and small embeddings."""
    return GrafeoPropertyGraphStore(
        embedding_dimensions=DIMS,
        embedding_metric="cosine",
        dedup_threshold=0.9,
    )


def _raw_props(store: GrafeoPropertyGraphStore, li_id: str) -> dict:
    """Get raw Grafeo node properties (including internal keys)."""
    gid = store._id_cache[li_id]
    return store.client.get_node(gid).properties()


class TestDedup:
    def test_disabled_by_default(self) -> None:
        store = GrafeoPropertyGraphStore()
        assert store.dedup_threshold is None

    def test_no_dedup_creates_separate_nodes(self) -> None:
        """Without dedup, similar embeddings create separate nodes."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        store.upsert_nodes([EntityNode(name="graph-db", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])])
        store.upsert_nodes(
            [EntityNode(name="graph-databases", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        all_nodes = store.get()
        assert len(all_nodes) == 2

    def test_dedup_merges_similar_entity(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """With dedup, a very similar embedding merges into the existing node."""
        dedup_store.upsert_nodes(
            [EntityNode(name="graph-db", label="entity", properties={"v": 1}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        # Second upsert with nearly identical embedding
        dedup_store.upsert_nodes(
            [
                EntityNode(
                    name="graph-databases",
                    label="entity",
                    properties={"v": 2, "extra": "new"},
                    embedding=[0.1, 0.8, 0.3, 0.5],
                )
            ]
        )

        # Only one Grafeo node should exist
        all_nodes = dedup_store.get()
        assert len(all_nodes) == 1
        # Properties merged from second upsert
        assert all_nodes[0].properties["v"] == 2
        assert all_nodes[0].properties["extra"] == "new"

    def test_dedup_does_not_merge_below_threshold(self) -> None:
        """With a very strict threshold, moderately similar nodes stay separate."""
        store = GrafeoPropertyGraphStore(
            embedding_dimensions=DIMS,
            embedding_metric="cosine",
            dedup_threshold=0.999,
        )
        store.upsert_nodes([EntityNode(name="A", label="entity", properties={}, embedding=[1.0, 0.0, 0.0, 0.0])])
        store.upsert_nodes([EntityNode(name="B", label="entity", properties={}, embedding=[0.95, 0.31, 0.0, 0.0])])
        all_nodes = store.get()
        assert len(all_nodes) == 2

    def test_dedup_only_within_same_label(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """Nodes with different labels are never deduped against each other."""
        dedup_store.upsert_nodes([EntityNode(name="A", label="concept", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])])
        dedup_store.upsert_nodes([EntityNode(name="B", label="tool", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])])
        all_nodes = dedup_store.get()
        assert len(all_nodes) == 2

    def test_dedup_skips_nodes_without_embedding(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """EntityNode without embedding creates a new node even if similar ones exist."""
        dedup_store.upsert_nodes(
            [EntityNode(name="graph-db", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        dedup_store.upsert_nodes([EntityNode(name="graph-databases", label="entity", properties={})])
        all_nodes = dedup_store.get()
        assert len(all_nodes) == 2

    def test_dedup_skips_chunk_nodes(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """ChunkNodes are never deduped."""
        dedup_store.upsert_nodes(
            [ChunkNode(id_="c1", text="Hello", label="text_chunk", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        dedup_store.upsert_nodes(
            [ChunkNode(id_="c2", text="Hello again", label="text_chunk", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        all_nodes = dedup_store.get()
        assert len(all_nodes) == 2

    def test_dedup_preserves_created_at(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """Merged node retains the original created_at timestamp."""
        dedup_store.upsert_nodes(
            [EntityNode(name="graph-db", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )
        original_id = next(iter(dedup_store._id_cache.values()))
        original_created = dedup_store.client.get_node(original_id).properties()["created_at"]

        dedup_store.upsert_nodes(
            [EntityNode(name="graph-databases", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])]
        )

        props = dedup_store.client.get_node(original_id).properties()
        assert props["created_at"] == original_created

    def test_runtime_threshold_change(self) -> None:
        """Setting dedup_threshold at runtime takes effect on next upsert."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        store.upsert_nodes([EntityNode(name="A", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])])

        # Enable dedup after first insert
        store.dedup_threshold = 0.9
        store.upsert_nodes([EntityNode(name="B", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])])
        all_nodes = store.get()
        assert len(all_nodes) == 1

    def test_dedup_caches_incoming_id(self, dedup_store: GrafeoPropertyGraphStore) -> None:
        """After dedup merge, the incoming node's ID maps to the existing Grafeo node."""
        first = EntityNode(name="graph-db", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])
        dedup_store.upsert_nodes([first])
        original_gid = dedup_store._id_cache[first.id]

        second = EntityNode(name="graph-databases", label="entity", properties={}, embedding=[0.1, 0.8, 0.3, 0.5])
        dedup_store.upsert_nodes([second])

        assert second.id in dedup_store._id_cache
        assert dedup_store._id_cache[second.id] == original_gid
