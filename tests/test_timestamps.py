from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from llama_index.core.graph_stores.types import ChunkNode, EntityNode

from grafeo_llamaindex import GrafeoPropertyGraphStore


def _raw_props(store: GrafeoPropertyGraphStore, li_id: str) -> dict:
    """Get raw Grafeo node properties (including internal keys)."""
    gid = store._id_cache[li_id]
    return store.client.get_node(gid).properties()


class TestTimestamps:
    def test_insert_sets_created_at(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])

        props = _raw_props(store, node.id)
        assert "created_at" in props
        # Valid ISO 8601
        datetime.fromisoformat(props["created_at"])

    def test_insert_sets_updated_at(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])

        props = _raw_props(store, node.id)
        assert "updated_at" in props
        datetime.fromisoformat(props["updated_at"])

    def test_created_at_equals_updated_at_on_insert(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])

        props = _raw_props(store, node.id)
        assert props["created_at"] == props["updated_at"]

    def test_update_preserves_created_at(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={"age": 30})
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 6, 1, tzinfo=UTC)

        with patch("grafeo_llamaindex.property_graph_store.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            store.upsert_nodes([node])

        original_created = _raw_props(store, node.id)["created_at"]

        with patch("grafeo_llamaindex.property_graph_store.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            updated = EntityNode(name="Alice", label="person", properties={"age": 31})
            store.upsert_nodes([updated])

        props = _raw_props(store, node.id)
        assert props["created_at"] == original_created

    def test_update_changes_updated_at(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={"age": 30})
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 6, 1, tzinfo=UTC)

        with patch("grafeo_llamaindex.property_graph_store.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            store.upsert_nodes([node])

        original_updated = _raw_props(store, node.id)["updated_at"]

        with patch("grafeo_llamaindex.property_graph_store.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            updated = EntityNode(name="Alice", label="person", properties={"age": 31})
            store.upsert_nodes([updated])

        props = _raw_props(store, node.id)
        assert props["updated_at"] != original_updated
        assert props["updated_at"] == t2.isoformat()

    def test_timestamps_not_in_labelled_node_properties(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={"age": 30})
        store.upsert_nodes([node])

        results = store.get(ids=[node.id])
        assert len(results) == 1
        assert "created_at" not in results[0].properties
        assert "updated_at" not in results[0].properties

    def test_batch_shares_timestamp(self, store: GrafeoPropertyGraphStore) -> None:
        nodes = [
            EntityNode(name="Alice", label="person", properties={}),
            EntityNode(name="Bob", label="person", properties={}),
            EntityNode(name="Carol", label="person", properties={}),
        ]
        store.upsert_nodes(nodes)

        timestamps = {_raw_props(store, n.id)["updated_at"] for n in nodes}
        assert len(timestamps) == 1

    def test_chunk_node_timestamps(self, store: GrafeoPropertyGraphStore) -> None:
        node = ChunkNode(id_="chunk-1", text="Hello", label="text_chunk", properties={})
        store.upsert_nodes([node])

        props = _raw_props(store, "chunk-1")
        assert "created_at" in props
        assert "updated_at" in props
        datetime.fromisoformat(props["created_at"])
