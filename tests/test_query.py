from __future__ import annotations

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore
from grafeo_llamaindex.property_graph_store import _validate_key


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

    def test_by_properties_of_any_type(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [EntityNode(name="X", properties={"flag": True, "score": 1.5, "tags": ["a", "b"]}), EntityNode(name="Y")]
        )
        assert [n.id for n in store.get(properties={"flag": True})] == ["X"]
        assert [n.id for n in store.get(properties={"score": 1.5})] == ["X"]
        assert [n.id for n in store.get(properties={"tags": ["a", "b"]})] == ["X"]
        assert store.get(properties={"flag": None}) == []  # used to fail with "Undefined variable 'None'"


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

    def test_relation_names_as_given_or_normalized(self, populated_store: GrafeoPropertyGraphStore) -> None:
        assert len(populated_store.get_triplets(relation_names=["works at"])) == 1
        assert len(populated_store.get_triplets(relation_names=["WORKS_AT"])) == 1

    def test_limit_covers_all_relation_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        """The limit used to apply per relation name, so two names could return two triplets."""
        triplets = populated_store.get_triplets(relation_names=["KNOWS", "WORKS_AT"], limit=1)
        assert len(triplets) == 1

    def test_limit_applies_after_property_filter(self, store: GrafeoPropertyGraphStore) -> None:
        """The property filter used to run in Python after LIMIT, dropping matches."""
        store.upsert_nodes([EntityNode(name=f"n{i}", properties={"group": "x"}) for i in range(10)])
        store.upsert_nodes([EntityNode(name="target", properties={"group": "y"})])
        store.upsert_relations([Relation(label="R", source_id=f"n{i}", target_id=f"n{i + 1}") for i in range(9)])
        store.upsert_relations([Relation(label="R", source_id="n9", target_id="target")])
        triplets = store.get_triplets(properties={"group": "y"}, limit=1)
        assert [(t[0].id, t[2].id) for t in triplets] == [("n9", "target")]

    def test_combined_filters(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["Bob"], relation_names=["KNOWS"], ids=["Alice"])
        assert [(t[0].id, t[1].label, t[2].id) for t in triplets] == [("Alice", "KNOWS", "Bob")]


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
    """Values reach Grafeo as query parameters; only property keys are validated and inlined."""

    @pytest.mark.parametrize(
        "name",
        [
            "O'Brien",
            'say "hi"',
            "a\\b",
            "line\nbreak",
            "nul\x00byte",
            "rtl\u202eoverride",
            "50$ $name",
            "') DETACH DELETE n //",
        ],
        ids=["quote", "double_quote", "backslash", "newline", "nul", "bidi", "dollar", "injection"],
    )
    def test_value_roundtrip_through_queries(self, store: GrafeoPropertyGraphStore, name: str) -> None:
        store.upsert_nodes([EntityNode(name=name, label="person"), EntityNode(name="Other", label="person")])
        store.upsert_relations([Relation(label="KNOWS", source_id=name, target_id="Other")])

        found = store.get(properties={"name": name})
        assert len(found) == 1
        assert isinstance(found[0], EntityNode)
        assert found[0].name == name
        assert len(store.get_triplets(entity_names=[name])) == 1

        store.delete(entity_names=[name])
        assert store.node_count == 1
        assert store.edge_count == 0

    def test_keyword_property_keys(self, store: GrafeoPropertyGraphStore) -> None:
        """Keys that are query keywords (``group``) used to be a syntax error in the generated query."""
        store.upsert_nodes(
            [EntityNode(name="A", properties={"group": "x"}), EntityNode(name="B", properties={"group": "y"})]
        )
        store.upsert_relations([Relation(label="R", source_id="A", target_id="B")])
        assert [n.id for n in store.get(properties={"group": "x"})] == ["A"]
        assert len(store.get_triplets(properties={"group": "y"})) == 1
        store.delete(properties={"group": "x"})
        assert [n.id for n in store.get()] == ["B"]

    def test_validate_key_valid(self) -> None:
        assert _validate_key("name") == "name"
        assert _validate_key("_private") == "_private"

    def test_validate_key_any_text(self) -> None:
        # LlamaIndex metadata keys may hold spaces or hyphens; they are quoted in queries.
        assert _validate_key("file name") == "file name"
        assert _validate_key("page-label") == "page-label"

    @pytest.mark.parametrize("key", ["", "a`b"])
    def test_validate_key_invalid(self, key: str) -> None:
        with pytest.raises(ValueError, match="Invalid property key"):
            _validate_key(key)

    def test_filter_on_key_with_space(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="A", label="doc", properties={"file name": "a.txt"}),
                EntityNode(name="B", label="doc", properties={"file name": "b.txt"}),
            ]
        )
        assert [n.id for n in store.get(properties={"file name": "a.txt"})] == ["A"]
        store.delete(properties={"file name": "a.txt"})
        assert [n.id for n in store.get()] == ["B"]

    def test_get_with_special_chars(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="O'Brien", label="person", properties={})
        store.upsert_nodes([node])
        results = store.get(properties={"name": "O'Brien"})
        assert len(results) == 1
        assert results[0].id == "O'Brien"
