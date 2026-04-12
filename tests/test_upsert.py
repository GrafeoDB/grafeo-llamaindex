from __future__ import annotations

import time
import warnings

from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore


class TestUpsertNodes:
    def test_entity_node(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={"age": 30})
        store.upsert_nodes([node])

        results = store.get(ids=[node.id])
        assert len(results) == 1
        assert isinstance(results[0], EntityNode)
        assert results[0].name == "Alice"
        assert results[0].label == "person"
        assert results[0].properties["age"] == 30

    def test_chunk_node(self, store: GrafeoPropertyGraphStore) -> None:
        node = ChunkNode(id_="chunk-1", text="Hello world", label="text_chunk", properties={"source": "test"})
        store.upsert_nodes([node])

        results = store.get(ids=["chunk-1"])
        assert len(results) == 1
        assert isinstance(results[0], ChunkNode)
        assert results[0].text == "Hello world"
        assert results[0].properties["source"] == "test"

    def test_update_existing(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={"age": 30})
        store.upsert_nodes([node])

        updated = EntityNode(name="Alice", label="person", properties={"age": 31})
        store.upsert_nodes([updated])

        results = store.get(ids=[node.id])
        assert len(results) == 1
        assert results[0].properties["age"] == 31

    def test_multiple_nodes(self, store: GrafeoPropertyGraphStore) -> None:
        nodes = [
            EntityNode(name="Alice", label="person", properties={}),
            EntityNode(name="Bob", label="person", properties={}),
            EntityNode(name="Acme", label="company", properties={}),
        ]
        store.upsert_nodes(nodes)

        all_nodes = store.get()
        assert len(all_nodes) == 3

    def test_label_normalization(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="X", label="my-custom label", properties={})
        store.upsert_nodes([node])

        results = store.get(ids=[node.id])
        assert len(results) == 1
        # The LlamaIndex label is preserved in properties, Grafeo label is sanitized
        assert results[0].label == "my-custom label"

    def test_populates_id_cache(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])

        assert node.id in store._id_cache

    def test_populates_name_cache(self, store: GrafeoPropertyGraphStore) -> None:
        node = EntityNode(name="Alice", label="person", properties={})
        store.upsert_nodes([node])

        assert "Alice" in store._name_cache


class TestUpsertRelations:
    def test_basic_relation(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["Alice"])
        assert len(triplets) >= 1

        labels = {t[1].label for t in triplets}
        assert "KNOWS" in labels

    def test_edge_type_normalization(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["Bob"])
        labels = {t[1].label for t in triplets}
        # "works at" → "WORKS_AT"
        assert "WORKS_AT" in labels

    def test_relation_properties(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.get_triplets(entity_names=["Alice"])
        knows_triplets = [t for t in triplets if t[1].label == "KNOWS"]
        assert len(knows_triplets) == 1
        assert knows_triplets[0][1].properties["since"] == 2020

    def test_missing_source_node(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="Bob", label="person", properties={})])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store.upsert_relations(
                [
                    Relation(label="KNOWS", source_id="NonExistent", target_id="Bob", properties={}),
                ]
            )
        # Relation skipped with a warning
        assert len(w) == 1
        assert "NonExistent" in str(w[0].message)
        triplets = store.get_triplets()
        assert len(triplets) == 0


class TestRelationUpsertDedup:
    """Upserting the same (source, type, target) twice should update, not duplicate."""

    def test_duplicate_relation_updates_not_duplicates(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}),
                EntityNode(name="Bob", label="person", properties={}),
            ]
        )
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"v": 1})])
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"v": 2})])

        assert store.edge_count == 1
        triplets = store.get_triplets(entity_names=["Alice"])
        assert len(triplets) == 1
        assert triplets[0][1].properties["v"] == 2

    def test_different_types_kept_separate(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}),
                EntityNode(name="Bob", label="person", properties={}),
            ]
        )
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={})])
        store.upsert_relations([Relation(label="LIKES", source_id="Alice", target_id="Bob", properties={})])

        assert store.edge_count == 2

    def test_different_targets_kept_separate(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}),
                EntityNode(name="Bob", label="person", properties={}),
                EntityNode(name="Carol", label="person", properties={}),
            ]
        )
        store.upsert_relations(
            [
                Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={}),
                Relation(label="KNOWS", source_id="Alice", target_id="Carol", properties={}),
            ]
        )

        assert store.edge_count == 2


class TestRelationTimestamps:
    """Edge timestamps: created_at preserved on update, updated_at refreshed."""

    def test_created_at_preserved_on_update(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}),
                EntityNode(name="Bob", label="person", properties={}),
            ]
        )
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={})])

        # Read raw edge to check internal timestamps
        edge = store.client.get_edge(0)
        created = edge.properties()["created_at"]

        time.sleep(0.01)  # ensure wall clock advances
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"v": 2})])

        edge2 = store.client.get_edge(0)
        assert edge2.properties()["created_at"] == created
        assert edge2.properties()["updated_at"] > created

    def test_timestamps_not_leaked_to_triplets(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes(
            [
                EntityNode(name="Alice", label="person", properties={}),
                EntityNode(name="Bob", label="person", properties={}),
            ]
        )
        store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={})])

        triplets = store.get_triplets(entity_names=["Alice"])
        assert len(triplets) == 1
        rel_props = triplets[0][1].properties
        assert "created_at" not in rel_props
        assert "updated_at" not in rel_props
