from __future__ import annotations

from pathlib import Path

import grafeo

from grafeo_llamaindex import GrafeoPropertyGraphStore


class TestInit:
    def test_in_memory(self) -> None:
        store = GrafeoPropertyGraphStore()
        assert store.supports_structured_queries is True
        assert store.supports_vector_queries is True

    def test_persistent(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        store = GrafeoPropertyGraphStore(db_path=db_path)
        assert store._db.is_persistent

    def test_client_property(self, store: GrafeoPropertyGraphStore) -> None:
        assert isinstance(store.client, grafeo.GrafeoDB)

    def test_property_indices_created(self, store: GrafeoPropertyGraphStore) -> None:
        assert store._db.has_property_index("li_id")
        assert store._db.has_property_index("name")


class TestSchema:
    def test_empty(self, store: GrafeoPropertyGraphStore) -> None:
        schema = store.get_schema()
        assert isinstance(schema, dict)

    def test_after_insert(self, populated_store: GrafeoPropertyGraphStore) -> None:
        schema = populated_store.get_schema()
        label_names = [entry["name"] for entry in schema.get("labels", [])]
        assert "person" in label_names
        assert "company" in label_names

    def test_schema_str(self, populated_store: GrafeoPropertyGraphStore) -> None:
        schema_str = populated_store.get_schema_str()
        assert "Node labels:" in schema_str
        assert "Edge types:" in schema_str

    def test_schema_cache(self, populated_store: GrafeoPropertyGraphStore) -> None:
        schema1 = populated_store.get_schema()
        schema2 = populated_store.get_schema()
        assert schema1 is schema2  # same object = cached

    def test_schema_refresh(self, populated_store: GrafeoPropertyGraphStore) -> None:
        schema1 = populated_store.get_schema()
        schema2 = populated_store.get_schema(refresh=True)
        assert schema1 is not schema2  # different object after refresh


class TestPersist:
    def test_save_in_memory(self, store: GrafeoPropertyGraphStore, tmp_path: Path) -> None:
        from llama_index.core.graph_stores.types import EntityNode

        store.upsert_nodes([EntityNode(name="Alice", label="person", properties={})])
        persist_path = str(tmp_path / "saved.db")
        store.persist(persist_path)
        assert Path(persist_path).exists()


class TestClose:
    def test_close(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.close()  # should not raise

    def test_context_manager(self) -> None:
        with GrafeoPropertyGraphStore() as store:
            assert isinstance(store, GrafeoPropertyGraphStore)
            assert isinstance(store.client, grafeo.GrafeoDB)
