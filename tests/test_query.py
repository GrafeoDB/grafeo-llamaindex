from __future__ import annotations

import pytest
from llama_index.core.graph_stores.types import EntityNode

from grafeo_llamaindex import GrafeoPropertyGraphStore
from grafeo_llamaindex.property_graph_store import _escape, _validate_key


class TestGet:
    def test_by_ids(self, populated_store: GrafeoPropertyGraphStore) -> None:
        alice = populated_store.get(ids=["Alice"])
        assert len(alice) == 1
        assert isinstance(alice[0], EntityNode)
        assert alice[0].id == "Alice"

    def test_by_ids_missing(self, populated_store: GrafeoPropertyGraphStore) -> None:
        results = populated_store.get(ids=["NonExistent"])
        assert results == []

    def test_by_properties(self, populated_store: GrafeoPropertyGraphStore) -> None:
        results = populated_store.get(properties={"age": 25})
        assert len(results) == 1
        assert results[0].id == "Bob"

    def test_all(self, populated_store: GrafeoPropertyGraphStore) -> None:
        results = populated_store.get()
        assert len(results) == 3

    def test_empty_store(self, store: GrafeoPropertyGraphStore) -> None:
        assert store.get() == []


class TestGetTriplets:
    def test_all(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets()
        assert len(triplets) == 2

    def test_by_entity_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["Alice"])
        assert len(triplets) >= 1
        # Alice is involved in the KNOWS relation
        names = {t[0].id for t in triplets} | {t[2].id for t in triplets}
        assert "Alice" in names

    def test_by_relation_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(relation_names=["KNOWS"])
        assert len(triplets) == 1
        assert triplets[0][1].label == "KNOWS"

    def test_empty(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["NonExistent"])
        assert triplets == []

    def test_limit(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(limit=1)
        assert len(triplets) == 1


class TestGetRelMap:
    def test_empty_input(self, populated_store: GrafeoPropertyGraphStore) -> None:
        assert populated_store.get_rel_map([]) == []

    def test_single_hop(self, populated_store: GrafeoPropertyGraphStore) -> None:
        alice = populated_store.get(ids=["Alice"])
        triplets = populated_store.get_rel_map(alice, depth=1)
        assert len(triplets) >= 1

    def test_deduplication(self, populated_store: GrafeoPropertyGraphStore) -> None:
        alice = populated_store.get(ids=["Alice"])
        triplets = populated_store.get_rel_map(alice, depth=2)
        edge_keys = [f"{t[0].id}-{t[1].label}-{t[2].id}" for t in triplets]
        assert len(edge_keys) == len(set(edge_keys))

    def test_ignore_rels(self, populated_store: GrafeoPropertyGraphStore) -> None:
        alice = populated_store.get(ids=["Alice"])
        triplets = populated_store.get_rel_map(alice, depth=2, ignore_rels=["KNOWS"])
        labels = {t[1].label for t in triplets}
        assert "KNOWS" not in labels


class TestStructuredQuery:
    def test_gql(self, populated_store: GrafeoPropertyGraphStore) -> None:
        results = populated_store.structured_query("MATCH (n:person) RETURN n")
        assert len(results) >= 2

    def test_empty_result(self, populated_store: GrafeoPropertyGraphStore) -> None:
        results = populated_store.structured_query("MATCH (n:nonexistent) RETURN n")
        assert results == []


class TestDelete:
    def test_by_ids(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(ids=["Alice"])
        assert populated_store.get(ids=["Alice"]) == []

    def test_by_entity_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(entity_names=["Bob"])
        assert populated_store.get(ids=["Bob"]) == []

    def test_by_relation_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(relation_names=["KNOWS"])
        triplets = populated_store.get_triplets(relation_names=["KNOWS"])
        assert triplets == []

    def test_evicts_id_cache(self, populated_store: GrafeoPropertyGraphStore) -> None:
        assert populated_store._find_node_id("Alice") is not None
        populated_store.delete(ids=["Alice"])
        assert "Alice" not in populated_store._id_cache

    def test_evicts_name_cache(self, populated_store: GrafeoPropertyGraphStore) -> None:
        assert populated_store._find_node_by_name("Bob") is not None
        populated_store.delete(entity_names=["Bob"])
        assert "Bob" not in populated_store._name_cache

    def test_nonexistent_is_noop(self, store: GrafeoPropertyGraphStore) -> None:
        store.delete(ids=["nothing"])
        store.delete(entity_names=["nothing"])
        store.delete(relation_names=["nothing"])


class TestSanitization:
    def test_escape_single_quote(self) -> None:
        assert _escape("O'Brien") == "O\\'Brien"

    def test_escape_double_quote_passthrough(self) -> None:
        assert _escape('say "hi"') == 'say "hi"'

    def test_escape_backslash(self) -> None:
        assert _escape("a\\b") == "a\\\\b"

    def test_validate_key_valid(self) -> None:
        assert _validate_key("name") == "name"
        assert _validate_key("_private") == "_private"

    def test_validate_key_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid property key"):
            _validate_key("not valid")

    def test_get_with_special_chars(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="O'Brien", label="person", properties={})
        store.upsert_nodes([node])
        results = store.get(properties={"name": "O'Brien"})
        assert len(results) == 1
        assert results[0].id == "O'Brien"
