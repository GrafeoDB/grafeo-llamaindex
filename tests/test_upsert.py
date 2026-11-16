from __future__ import annotations

import time
import warnings
from pathlib import Path

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

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


RELATION_LABELS = [
    "works for/at (part-time)",
    "Alice's friend",
    "is-a.k.a",
    "relates:to",
    "gr\u00f6\u00dfer als",
    "x]->(y) //",
]


class TestRelationLabels:
    """LLM-extracted relation labels can hold any text; they used to break the generated queries."""

    def test_upsert_update_and_read(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="A"), EntityNode(name="B")])
        for label in RELATION_LABELS:
            store.upsert_relations([Relation(label=label, source_id="A", target_id="B", properties={"v": 1})])
            store.upsert_relations([Relation(label=label, source_id="A", target_id="B", properties={"v": 2})])
        assert store.edge_count == len(RELATION_LABELS)
        for label in RELATION_LABELS:
            triplets = store.get_triplets(relation_names=[label])
            assert len(triplets) == 1, label
            assert triplets[0][1].properties["v"] == 2


class TestEndpointResolution:
    def test_first_node_resolved_by_name(self, store: GrafeoPropertyGraphStore) -> None:
        """Grafeo ID 0 is a valid node: resolving it by name must not fall through as falsy.

        An entity whose name holds a double quote has a LlamaIndex ID that differs from its name,
        so the fallback lookup by ID found nothing and the relation was skipped with a warning.
        """
        quoted = EntityNode(name='say "hi"', label="phrase")
        assert quoted.id != quoted.name
        store.upsert_nodes([quoted, EntityNode(name="B")])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            store.upsert_relations([Relation(label="R", source_id=quoted.name, target_id="B")])
        assert store.edge_count == 1


class TestEmbeddingDimensions:
    def test_wrong_dimensions_rejected_before_writing(self) -> None:
        """A mismatch used to be stored, after which the label's vector index could not be built
        and vector_query() silently returned nothing for that label, even for correct embeddings."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=8)
        with pytest.raises(ValueError, match="4 dimensions"):
            store.upsert_nodes(
                [
                    EntityNode(name="ok", embedding=[0.1] * 8),
                    EntityNode(name="bad", embedding=[0.1, 0.2, 0.3, 0.4]),
                ]
            )
        assert store.node_count == 0

        store.upsert_nodes([EntityNode(name="ok", embedding=[0.1] * 8)])
        nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=[0.1] * 8, similarity_top_k=1))
        assert [n.id for n in nodes] == ["ok"]

    def test_wrong_dimensions_with_dedup(self) -> None:
        """With dedup on, a mismatched embedding reached Grafeo's vector search, which panics."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=4, dedup_threshold=0.9)
        store.upsert_nodes([EntityNode(name="a", embedding=[0.1, 0.2, 0.3, 0.4])])
        with pytest.raises(ValueError, match="dimensions"):
            store.upsert_nodes([EntityNode(name="b", embedding=[0.1, 0.2])])
        assert store.node_count == 1

    def test_nodes_without_embedding_are_fine(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=4)
        store.upsert_nodes([EntityNode(name="a"), ChunkNode(id_="c", text="t", embedding=[])])
        assert store.node_count == 2


class TestUpdatePaths:
    def test_update_keeps_relations(self, populated_store: GrafeoPropertyGraphStore) -> None:
        populated_store.upsert_nodes([EntityNode(name="Bob", label="person", properties={"age": 26})])
        assert populated_store.node_count == 3
        assert populated_store.edge_count == 2
        assert populated_store.get(ids=["Bob"])[0].properties["age"] == 26

    def test_update_embedding_moves_vector_result(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=4)
        store.upsert_nodes(
            [EntityNode(name="a", embedding=[1.0, 0.0, 0.0, 0.0]), EntityNode(name="b", embedding=[0.0, 1.0, 0.0, 0.0])]
        )
        store.upsert_nodes([EntityNode(name="a", embedding=[0.0, 0.0, 1.0, 0.0])])
        nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=[0.0, 0.0, 1.0, 0.0], similarity_top_k=1))
        assert [n.id for n in nodes] == ["a"]
        assert store.node_count == 2

    def test_chunk_text_update(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([ChunkNode(id_="c1", text="old")])
        store.upsert_nodes([ChunkNode(id_="c1", text="new")])
        [chunk] = store.get(ids=["c1"])
        assert isinstance(chunk, ChunkNode)
        assert chunk.text == "new"


class TestChunkKind:
    """A ChunkNode stays a chunk whatever its label: it is marked as one, not recognized by label."""

    def test_custom_chunk_label_round_trips(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([ChunkNode(text="page one", id_="p1", label="page")])
        [node] = store.get(ids=["p1"])
        assert isinstance(node, ChunkNode)
        assert (node.label, node.text) == ("page", "page one")

    def test_vector_query_skips_chunks_sharing_a_label(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=4)
        store.upsert_nodes(
            [
                ChunkNode(text="t", id_="c1", label="doc", embedding=[1.0, 0.0, 0.0, 0.0]),
                EntityNode(name="e", label="doc", embedding=[0.9, 0.1, 0.0, 0.0]),
            ]
        )
        nodes, _scores = store.vector_query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2))
        assert [n.id for n in nodes] == ["e"]

    def test_dedup_never_merges_into_a_chunk(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=4, dedup_threshold=0.9)
        store.upsert_nodes([ChunkNode(text="t", id_="c1", label="doc", embedding=[1.0, 0.0, 0.0, 0.0])])
        store.upsert_nodes([EntityNode(name="e", label="doc", embedding=[1.0, 0.0, 0.0, 0.0])])
        assert {type(n).__name__ for n in store.get()} == {"ChunkNode", "EntityNode"}

    def test_chunk_label_known_after_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "kg.grafeo")
        with GrafeoPropertyGraphStore(db_path=path, embedding_dimensions=4) as store:
            store.upsert_nodes(
                [
                    ChunkNode(text="t", id_="c1", label="doc", embedding=[1.0, 0.0, 0.0, 0.0]),
                    EntityNode(name="e", label="doc", embedding=[0.9, 0.1, 0.0, 0.0]),
                ]
            )
        with GrafeoPropertyGraphStore(db_path=path, embedding_dimensions=4) as store:
            query = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
            assert [n.id for n in store.vector_query(query)[0]] == ["e"]
            assert isinstance(store.get(ids=["c1"])[0], ChunkNode)


class TestRelabel:
    """Upserting a node again with another label moves it to that label."""

    def test_relabel_updates_grafeo_label(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="Scrooge", label="PERSON")])
        store.upsert_nodes([EntityNode(name="Scrooge", label="CHARACTER")])
        assert store.get(ids=["Scrooge"])[0].label == "CHARACTER"
        assert [entry["name"] for entry in store.get_schema(refresh=True)["labels"]] == ["CHARACTER"]
        assert store.structured_query("MATCH (n:CHARACTER) RETURN n.name AS name") == [{"name": "Scrooge"}]

    def test_relabel_keeps_vector_search(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=4)
        store.upsert_nodes([EntityNode(name="Scrooge", label="PERSON", embedding=[0.1, 0.8, 0.3, 0.5])])
        store.upsert_nodes([EntityNode(name="Scrooge", label="CHARACTER")])  # no embedding this time
        query = VectorStoreQuery(query_embedding=[0.1, 0.8, 0.3, 0.5], similarity_top_k=1)
        nodes, _scores = store.vector_query(query)
        assert [(n.id, n.label) for n in nodes] == [("Scrooge", "CHARACTER")]


class TestCaseInsensitiveNames:
    """LlamaIndex's default extractor capitalises names ("Gavin belson"); they match entities written
    with their real case ("Gavin Belson") instead of becoming duplicates."""

    @staticmethod
    def _store(case_insensitive_names: bool = True) -> GrafeoPropertyGraphStore:
        store = GrafeoPropertyGraphStore(case_insensitive_names=case_insensitive_names)
        store.upsert_nodes(
            [
                EntityNode(name="Gavin Belson", label="Person", properties={"pledge": "Tethics"}),
                EntityNode(name="Hooli", label="Company"),
            ]
        )
        return store

    def test_extracted_name_merges_into_existing_entity(self) -> None:
        store = self._store()
        store.upsert_nodes([EntityNode(name="Gavin belson", label="entity", properties={"triplet_source_id": "c1"})])
        [gavin] = [n for n in store.get() if n.id.lower() == "gavin belson"]
        assert (gavin.id, gavin.label, gavin.properties["pledge"]) == ("Gavin Belson", "Person", "Tethics")
        assert [n.id for n in store.get(ids=["Gavin belson"])] == ["Gavin Belson"]

    def test_relation_endpoints_match_any_case(self) -> None:
        store = self._store()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            store.upsert_relations([Relation(label="WAS_CEO_OF", source_id="Gavin belson", target_id="hooli")])
        assert [(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()] == [
            ("Gavin Belson", "WAS_CEO_OF", "Hooli")
        ]
        assert len(store.get_triplets(entity_names=["GAVIN BELSON"])) == 1

    def test_delete_by_name_matches_any_case(self) -> None:
        store = self._store()
        store.delete(entity_names=["hooli"])
        assert [n.id for n in store.get()] == ["Gavin Belson"]

    def test_option_off_keeps_case(self) -> None:
        store = self._store(case_insensitive_names=False)
        store.upsert_nodes([EntityNode(name="Gavin belson", label="entity")])
        assert sorted(n.id for n in store.get()) == ["Gavin Belson", "Gavin belson", "Hooli"]

    def test_chunks_are_not_matched(self) -> None:
        store = self._store()
        store.upsert_nodes([ChunkNode(text="t", id_="hooli")])
        assert sorted(type(n).__name__ for n in store.get()) == ["ChunkNode", "EntityNode", "EntityNode"]

    def test_match_survives_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "kg.grafeo")
        with GrafeoPropertyGraphStore(db_path=path) as store:
            store.upsert_nodes([EntityNode(name="Gavin Belson", label="Person")])
        with GrafeoPropertyGraphStore(db_path=path) as store:
            store.upsert_nodes([EntityNode(name="Gavin belson", label="entity")])
            assert [n.id for n in store.get()] == ["Gavin Belson"]

    def test_schema_hides_the_folded_name(self) -> None:
        assert "li_name_key" not in self._store().get_schema_str(refresh=True)
