from __future__ import annotations

from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4


def _edges(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


class TestDeleteRemovesAttachedRelations:
    """delete() of a node also deletes its relations, in both directions (grafeo >=0.5.44 refuses
    to delete a node that still has edges, which made these calls raise GrafeoError)."""

    def test_by_ids(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(ids=["Bob"])  # Bob has an incoming and an outgoing relation
        assert populated_store.node_count == 2
        assert populated_store.edge_count == 0
        assert {n.id for n in populated_store.get()} == {"Alice", "Acme"}

    def test_by_entity_names(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(entity_names=["Alice"])
        assert populated_store.node_count == 2
        assert _edges(populated_store) == {("Bob", "WORKS_AT", "Acme")}

    def test_by_properties(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(properties={"industry": "tech"})
        assert {n.id for n in populated_store.get()} == {"Alice", "Bob"}
        assert _edges(populated_store) == {("Alice", "KNOWS", "Bob")}

    def test_chunk_with_mentions(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                ChunkNode(id_="chunk-1", text="Alice works at Acme."),
                EntityNode(name="Alice", label="person"),
                EntityNode(name="Acme", label="company"),
            ]
        )
        store.upsert_relations(
            [
                Relation(label="MENTIONS", source_id="Alice", target_id="chunk-1"),
                Relation(label="MENTIONS", source_id="Acme", target_id="chunk-1"),
                Relation(label="WORKS_AT", source_id="Alice", target_id="Acme"),
            ]
        )
        store.delete(ids=["chunk-1"])
        assert store.get(ids=["chunk-1"]) == []
        assert _edges(store) == {("Alice", "WORKS_AT", "Acme")}

    def test_overlapping_criteria(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(ids=["Alice", "Bob"], entity_names=["Bob"], properties={"age": 30})
        assert {n.id for n in populated_store.get()} == {"Acme"}
        assert populated_store.edge_count == 0

    def test_nodes_and_relation_names_together(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(ids=["Acme"], relation_names=["knows"])
        assert populated_store.node_count == 2
        assert populated_store.edge_count == 0

    def test_all_nodes_with_the_name(self, store: GrafeoPropertyGraphStore) -> None:
        """Nodes written outside the store can share a name: delete(entity_names) removes all of them."""
        store.upsert_nodes([EntityNode(name="Apple", label="company")])
        store.client.create_node(labels=["fruit"], properties={"name": "Apple", "li_id": "apple-fruit"})
        store.delete(entity_names=["Apple"])
        assert store.node_count == 0

    def test_deleted_node_leaves_vector_results(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name="keep", embedding=[1.0, 0.0, 0.0, 0.0]),
                EntityNode(name="drop", embedding=[1.0, 0.1, 0.0, 0.0]),
            ]
        )
        store.upsert_relations([Relation(label="R", source_id="drop", target_id="keep")])
        store.delete(ids=["drop"])
        nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=[1.0, 0.1, 0.0, 0.0], similarity_top_k=5))
        assert [n.id for n in nodes] == ["keep"]

    def test_reupsert_after_delete(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(ids=["Alice"])
        populated_store.upsert_nodes([EntityNode(name="Alice", label="person", properties={"age": 40})])
        populated_store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob")])
        alice = populated_store.get(ids=["Alice"])
        assert len(alice) == 1
        assert alice[0].properties["age"] == 40
        assert ("Alice", "KNOWS", "Bob") in _edges(populated_store)
        assert populated_store.node_count == 3


class TestDeleteRelationNames:
    def test_normalized_label(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.delete(relation_names=["works at"])
        assert _edges(populated_store) == {("Alice", "KNOWS", "Bob")}
        assert populated_store.node_count == 3

    def test_label_with_special_characters(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="A"), EntityNode(name="B")])
        store.upsert_relations([Relation(label="works for/at (part-time)", source_id="A", target_id="B")])
        assert store.edge_count == 1
        store.delete(relation_names=["works for/at (part-time)"])
        assert store.edge_count == 0
        assert store.node_count == 2


class TestDeleteKeepsCachesConsistent:
    def test_dedup_aliases_evicted(self) -> None:
        """Two LlamaIndex IDs merged into one node by dedup: deleting one evicts both."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, dedup_threshold=0.9)
        store.upsert_nodes([EntityNode(name="graph-db", embedding=[0.1, 0.8, 0.3, 0.5])])
        store.upsert_nodes([EntityNode(name="graph-databases", embedding=[0.1, 0.8, 0.3, 0.5])])
        assert store.node_count == 1

        store.delete(ids=["graph-db"])  # the alias: the node's li_id is now "graph-databases"
        assert store.node_count == 0
        assert store._id_cache == {}
        assert store._name_cache == {}
        assert store.get(ids=["graph-databases"]) == []

    def test_delete_by_alias_and_reupsert(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS, dedup_threshold=0.9)
        store.upsert_nodes([EntityNode(name="graph-db", embedding=[0.1, 0.8, 0.3, 0.5])])
        store.upsert_nodes([EntityNode(name="graph-databases", embedding=[0.1, 0.8, 0.3, 0.5])])
        store.delete(entity_names=["graph-db"])
        store.upsert_nodes([EntityNode(name="graph-db", embedding=[0.9, 0.1, 0.0, 0.0])])
        assert [n.id for n in store.get()] == ["graph-db"]


class TestWritesOutsideTheStore:
    """Nodes deleted through structured_query() or client are not served from stale caches."""

    def test_upsert_after_structured_query_delete(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.structured_query("MATCH (n) DETACH DELETE n")
        # Before: GrafeoError "node 0 does not exist" from the cached Grafeo ID.
        populated_store.upsert_nodes([EntityNode(name="Alice", label="person", properties={"age": 1})])
        assert populated_store.node_count == 1
        assert populated_store.get(ids=["Alice"])[0].properties["age"] == 1

    def test_relations_after_client_delete(self, populated_store: GrafeoPropertyGraphStore) -> None:
        client = populated_store.client
        bob = client.find_nodes_by_property("li_id", "Bob")[0]
        client.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": bob})
        populated_store.upsert_nodes([EntityNode(name="Bob", label="person")])
        populated_store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob")])
        assert ("Alice", "KNOWS", "Bob") in _edges(populated_store)

    def test_get_after_client_recreate(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.structured_query("MATCH (n) WHERE n.li_id = 'Bob' DETACH DELETE n")
        populated_store.client.create_node(labels=["person"], properties={"li_id": "Bob", "name": "Bob"})
        assert [n.id for n in populated_store.get(ids=["Bob"])] == ["Bob"]
        populated_store.delete(ids=["Bob"])
        assert populated_store.get(ids=["Bob"]) == []
