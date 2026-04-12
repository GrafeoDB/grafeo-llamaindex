from __future__ import annotations

import warnings
from pathlib import Path

import grafeo
import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


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


class TestEmbeddingDimensionMismatch:
    """T3: Upsert nodes with one dimension, query with a different dimension."""

    def test_vector_query_wrong_dimensions_raises(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, embedding_metric="cosine")
        store.upsert_nodes([EntityNode(name="X", label="entity", properties={}, embedding=[0.1] * DIMS)])
        wrong_dims = [0.1] * (DIMS * 2)
        query = VectorStoreQuery(query_embedding=wrong_dims, similarity_top_k=1)
        with pytest.raises(ValueError, match="dimensions"):
            store.vector_query(query)


class TestMissingRelationEndpoints:
    """T4: upsert_relations with missing source/target emits a warning."""

    def test_missing_source_warns(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="Bob", label="person", properties={})])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store.upsert_relations([Relation(label="KNOWS", source_id="Ghost", target_id="Bob", properties={})])
        assert len(w) == 1
        assert "source_id='Ghost'" in str(w[0].message)

    def test_missing_target_warns(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="Alice", label="person", properties={})])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Ghost", properties={})])
        assert len(w) == 1
        assert "target_id='Ghost'" in str(w[0].message)

    def test_both_missing_warns(self, store: GrafeoPropertyGraphStore) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store.upsert_relations([Relation(label="KNOWS", source_id="A", target_id="B", properties={})])
        assert len(w) == 1
        assert "source_id='A'" in str(w[0].message)
        assert "target_id='B'" in str(w[0].message)


class TestSpecialCharacterProperties:
    """T5: Special characters in properties round-trip correctly."""

    @pytest.mark.parametrize(
        "name",
        [
            "O'Malley",
            'say "hello"',
            "back\\slash",
            "\u00e9\u00e0\u00fc",
            "line\nnewline",
        ],
        ids=["single_quote", "double_quote", "backslash", "unicode", "newline"],
    )
    def test_special_name_roundtrip(self, name: str) -> None:
        store = GrafeoPropertyGraphStore()
        node = EntityNode(name=name, label="entity", properties={})
        store.upsert_nodes([node])
        # Look up by the node's actual id (LlamaIndex may sanitize the id)
        results = store.get(ids=[node.id])
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, EntityNode)
        assert result.name == name


class TestSchemaCompleteness:
    """T6: Multi-label, multi-edge graph schema includes all labels and edge types."""

    def test_schema_str_includes_all_labels_and_edges(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="Person", properties={}),
                EntityNode(name="Acme", label="Company", properties={}),
                EntityNode(name="NYC", label="Location", properties={}),
            ]
        )
        store.upsert_relations(
            [
                Relation(label="KNOWS", source_id="Alice", target_id="Acme", properties={}),
                Relation(label="WORKS_AT", source_id="Alice", target_id="Acme", properties={}),
                Relation(label="LOCATED_IN", source_id="Acme", target_id="NYC", properties={}),
            ]
        )
        schema_str = store.get_schema_str()

        # All node labels present
        assert "Person" in schema_str
        assert "Company" in schema_str
        assert "Location" in schema_str

        # All edge types present
        assert "KNOWS" in schema_str
        assert "WORKS_AT" in schema_str
        assert "LOCATED_IN" in schema_str


class TestParamMapSubstitution:
    """structured_query() should substitute $param placeholders from param_map."""

    def test_string_param(self, populated_store: GrafeoPropertyGraphStore) -> None:
        rows = populated_store.structured_query(
            "MATCH (n) WHERE n.name = $name RETURN n",
            param_map={"name": "Alice"},
        )
        assert len(rows) == 1

    def test_int_param(self, populated_store: GrafeoPropertyGraphStore) -> None:
        rows = populated_store.structured_query(
            "MATCH (n) WHERE n.age = $age RETURN n",
            param_map={"age": 30},
        )
        assert len(rows) == 1

    def test_list_param(self, populated_store: GrafeoPropertyGraphStore) -> None:
        rows = populated_store.structured_query(
            "MATCH (n) WHERE n.name IN $names RETURN n",
            param_map={"names": ["Alice", "Bob"]},
        )
        assert len(rows) == 2

    def test_none_param(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name="X", label="entity", properties={})])
        # $val replaced with null, so WHERE n.name = null matches nothing
        rows = store.structured_query(
            "MATCH (n) WHERE n.name = $val RETURN n",
            param_map={"val": None},
        )
        assert len(rows) == 0

    def test_special_chars_escaped(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name="O'Malley", label="entity", properties={})])
        rows = store.structured_query(
            "MATCH (n) WHERE n.name = $name RETURN n",
            param_map={"name": "O'Malley"},
        )
        assert len(rows) == 1

    def test_no_param_map_unchanged(self, populated_store: GrafeoPropertyGraphStore) -> None:
        rows = populated_store.structured_query("MATCH (n) WHERE n.name = 'Alice' RETURN n")
        assert len(rows) == 1


class TestDeleteByProperties:
    """delete(properties=...) should remove nodes matching the given properties."""

    def test_delete_by_single_property(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={"role": "admin"}),
                EntityNode(name="Bob", label="person", properties={"role": "user"}),
            ]
        )
        store.delete(properties={"role": "admin"})
        assert store.node_count == 1
        remaining = store.get()
        assert isinstance(remaining[0], EntityNode)
        assert remaining[0].name == "Bob"

    def test_delete_by_properties_evicts_caches(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name="Alice", label="person", properties={"role": "admin"})])
        node_id = store.get()[0].id
        assert node_id in store._id_cache
        assert "Alice" in store._name_cache

        store.delete(properties={"role": "admin"})
        assert node_id not in store._id_cache
        assert "Alice" not in store._name_cache

    def test_delete_by_properties_no_match(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name="Alice", label="person", properties={"role": "admin"})])
        store.delete(properties={"role": "nobody"})
        assert store.node_count == 1


class TestCacheEviction:
    """Bidirectional cache eviction: deleting by ID also clears name cache and vice versa."""

    def test_delete_by_id_clears_name_cache(self) -> None:
        store = GrafeoPropertyGraphStore()
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])
        assert "Alice" in store._name_cache

        store.delete(ids=[node.id])
        assert "Alice" not in store._name_cache

    def test_delete_by_name_clears_id_cache(self) -> None:
        store = GrafeoPropertyGraphStore()
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])
        assert node.id in store._id_cache

        store.delete(entity_names=["Alice"])
        assert node.id not in store._id_cache


class TestGetTripletsPropertiesFilter:
    """get_triplets(properties=...) should filter triplets by node properties."""

    def test_filter_by_property(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(properties={"age": 30})
        # Alice has age=30, so triplets involving Alice should match
        assert len(triplets) >= 1
        names = {t[0].id for t in triplets} | {t[2].id for t in triplets}
        assert "Alice" in names

    def test_filter_by_string_property(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(properties={"industry": "tech"})
        assert len(triplets) >= 1
        names = {t[0].id for t in triplets} | {t[2].id for t in triplets}
        assert "Acme" in names

    def test_filter_by_nonexistent_property(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(properties={"role": "ghost"})
        assert len(triplets) == 0
