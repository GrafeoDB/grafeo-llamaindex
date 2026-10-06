"""get_rel_map() follows relations in both directions, checked against LlamaIndex's reference store."""

from __future__ import annotations

import pytest
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, PropertyGraphStore, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore

#   Carol -KNOWS-> Alice -KNOWS-> Bob -WORKS_AT-> Acme -LOCATED_IN-> Paris
#                                 Bob -LIKES-> Tea      Fund -INVESTS_IN-> Acme
GRAPH = [
    ("Alice", "KNOWS", "Bob"),
    ("Bob", "WORKS_AT", "Acme"),
    ("Acme", "LOCATED_IN", "Paris"),
    ("Bob", "LIKES", "Tea"),
    ("Fund", "INVESTS_IN", "Acme"),
    ("Carol", "KNOWS", "Alice"),
]
NAMES = sorted({name for source, _, target in GRAPH for name in (source, target)})

Edge = tuple[str, str, str]


def _load(store: PropertyGraphStore) -> PropertyGraphStore:
    store.upsert_nodes([EntityNode(name=name) for name in NAMES])
    store.upsert_relations([Relation(label=label, source_id=s, target_id=t) for s, label, t in GRAPH])
    return store


def _rel_map(
    store: PropertyGraphStore, seeds: list[str], depth: int, ignore_rels: list[str] | None = None, limit: int = 100
) -> list[Edge]:
    nodes = [EntityNode(name=name) for name in seeds]
    triplets = store.get_rel_map(nodes, depth=depth, limit=limit, ignore_rels=ignore_rels)
    return [(t[0].id, t[1].label, t[2].id) for t in triplets]


@pytest.fixture(scope="module")
def reference() -> PropertyGraphStore:
    return _load(SimplePropertyGraphStore())


@pytest.fixture(scope="module")
def grafeo_store() -> PropertyGraphStore:
    return _load(GrafeoPropertyGraphStore())


class TestSameAsReferenceStore:
    """The same triplets as SimplePropertyGraphStore, which matches a node as subject or object."""

    @pytest.mark.parametrize("seeds", [[name] for name in NAMES] + [["Paris", "Carol"], ["Tea", "Fund"]])
    def test_depth_1(self, reference: PropertyGraphStore, grafeo_store: PropertyGraphStore, seeds: list[str]) -> None:
        assert set(_rel_map(grafeo_store, seeds, 1)) == set(_rel_map(reference, seeds, 1))

    @pytest.mark.parametrize("seeds", [["Alice"], ["Fund"], ["Carol"], ["Alice", "Fund"]])
    def test_depth_2(self, reference: PropertyGraphStore, grafeo_store: PropertyGraphStore, seeds: list[str]) -> None:
        assert set(_rel_map(grafeo_store, seeds, 2)) == set(_rel_map(reference, seeds, 2))

    @pytest.mark.parametrize(
        ("seeds", "depth", "ignore_rels"),
        [
            (["Alice"], 2, ["LIKES"]),
            (["Bob"], 1, ["KNOWS"]),
            (["Fund"], 2, ["LOCATED_IN"]),
            (["Acme"], 1, ["WORKS_AT"]),
        ],
    )
    def test_ignore_rels(
        self,
        reference: PropertyGraphStore,
        grafeo_store: PropertyGraphStore,
        seeds: list[str],
        depth: int,
        ignore_rels: list[str],
    ) -> None:
        expected = set(_rel_map(reference, seeds, depth, ignore_rels))
        assert set(_rel_map(grafeo_store, seeds, depth, ignore_rels)) == expected


class TestBothDirections:
    def test_incoming_relations_of_a_target(self, grafeo_store: PropertyGraphStore) -> None:
        """Before 0.2.1 only outgoing relations were followed, so a target such as Paris had none."""
        assert set(_rel_map(grafeo_store, ["Paris"], 1)) == {("Acme", "LOCATED_IN", "Paris")}

    def test_second_hop_from_either_end(self, grafeo_store: PropertyGraphStore) -> None:
        """Undirected paths, as in Neo4jPropertyGraphStore's ``(e)-[r*1..depth]-(other)``.

        SimplePropertyGraphStore expands the second hop only from the targets of the first, so
        from Paris it stops at Acme -LOCATED_IN-> Paris.
        """
        assert set(_rel_map(grafeo_store, ["Paris"], 2)) == {
            ("Acme", "LOCATED_IN", "Paris"),
            ("Bob", "WORKS_AT", "Acme"),
            ("Fund", "INVESTS_IN", "Acme"),
        }

    def test_ignored_relations_are_not_followed(self, grafeo_store: PropertyGraphStore) -> None:
        """Reaching Bob needs Alice -KNOWS-> Bob, so nothing beyond it is returned."""
        assert _rel_map(grafeo_store, ["Alice"], 3, ignore_rels=["KNOWS"]) == []

    def test_each_relation_once_in_stored_direction(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="A"), EntityNode(name="B")])
        store.upsert_relations(
            [
                Relation(label="R", source_id="A", target_id="B"),
                Relation(label="R", source_id="B", target_id="A"),
                Relation(label="SELF", source_id="A", target_id="A"),
            ]
        )
        assert sorted(_rel_map(store, ["A", "B"], 2)) == [("A", "R", "B"), ("A", "SELF", "A"), ("B", "R", "A")]

    def test_nearest_hops_first_within_limit(self, grafeo_store: PropertyGraphStore) -> None:
        first_hop = set(_rel_map(grafeo_store, ["Bob"], 1))
        assert len(first_hop) == 3
        assert set(_rel_map(grafeo_store, ["Bob"], 3, limit=3)) == first_hop
        assert len(_rel_map(grafeo_store, ["Bob"], 3, limit=2)) == 2

    def test_whole_component_at_depth(self, grafeo_store: PropertyGraphStore) -> None:
        assert set(_rel_map(grafeo_store, ["Paris"], 4)) == set(GRAPH)
