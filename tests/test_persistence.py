from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4

# A database written by this store on grafeo 0.5.44 (the last 0.5 release), by running
# ``uv run --no-sources python tests/test_persistence.py`` with grafeo 0.5.44 installed.
# grafeo 0.6.0 migrates 0.5.x files to its new format on the first read-write open.
FIXTURE_0_5_44 = Path(__file__).parent / "data" / "store-grafeo-0.5.44.grafeo"

# Both on-disk layouts: a single .grafeo file, and the WAL directory any other path creates
# (deprecated from grafeo 0.6.0, still readable).
LAYOUTS = pytest.mark.parametrize("filename", ["kg.grafeo", "kg.db"], ids=["single_file", "wal_directory"])


def _vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIMS)]


def _populate(store: GrafeoPropertyGraphStore) -> None:
    """Alice-[KNOWS]->Bob-[WORKS_AT]->Acme, Alice-[MENTIONS]->chunk-1, all with embeddings."""
    store.upsert_nodes(
        [
            EntityNode(name="Alice", label="person", properties={"age": 30}, embedding=_vec(0)),
            EntityNode(name="Bob", label="person", properties={"age": 25}, embedding=_vec(1)),
            EntityNode(name="Acme", label="company", properties={"industry": "tech"}, embedding=_vec(2)),
            ChunkNode(id_="chunk-1", text="Alice knows Bob.", properties={"ref_doc_id": "doc-1"}, embedding=_vec(3)),
        ]
    )
    store.upsert_relations(
        [
            Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"since": 2020}),
            Relation(label="works at", source_id="Bob", target_id="Acme"),
            Relation(label="MENTIONS", source_id="Alice", target_id="chunk-1"),
        ]
    )


def _triplets(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


def _nearest(store: GrafeoPropertyGraphStore, i: int) -> str:
    nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=_vec(i), similarity_top_k=1))
    return nodes[0].id


class TestPersistenceRoundTrip:
    """T7: Create, populate, close, reopen, verify data and indexes survive."""

    def test_nodes_survive_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist.db")

        nodes = [EntityNode(name=f"entity_{i}", label="entity", properties={"idx": i}) for i in range(20)]
        relations = [
            Relation(
                label="LINKS_TO",
                source_id=f"entity_{i}",
                target_id=f"entity_{i + 1}",
                properties={},
            )
            for i in range(10)
        ]

        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(nodes)
        store.upsert_relations(relations)

        original_node_count = store.node_count
        original_edge_count = store.edge_count
        assert original_node_count == 20
        assert original_edge_count == 10

        store.close()

        # Reopen
        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        assert store2.node_count == original_node_count
        assert store2.edge_count == original_edge_count
        store2.close()

    def test_vector_search_survives_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist_vec.db")

        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name=f"node_{i}", label="entity", properties={}, embedding=[float(i) / 20] * DIMS)
                for i in range(20)
            ]
        )
        store.upsert_relations(
            [
                Relation(
                    label="LINKS_TO",
                    source_id=f"node_{i}",
                    target_id=f"node_{i + 1}",
                    properties={},
                )
                for i in range(10)
            ]
        )
        store.close()

        # Reopen: vector indexes are rediscovered automatically, no probe needed
        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)

        query = VectorStoreQuery(query_embedding=[0.5] * DIMS, similarity_top_k=5)
        nodes, scores = store2.vector_query(query)
        assert len(nodes) > 0
        assert len(scores) == len(nodes)
        store2.close()

    def test_vector_index_labels_rediscovered(self, tmp_path: Path) -> None:
        """_vector_indexed_labels is populated from existing indexes on reopen."""
        db_path = str(tmp_path / "rediscover.db")
        store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        store.upsert_nodes(
            [
                EntityNode(name="a", label="person", properties={}, embedding=[1.0, 0.0, 0.0, 0.0]),
                EntityNode(name="b", label="company", properties={}, embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        assert "person" in store._vector_indexed_labels
        assert "company" in store._vector_indexed_labels
        store.close()

        store2 = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS)
        assert "person" in store2._vector_indexed_labels
        assert "company" in store2._vector_indexed_labels
        store2.close()

    def test_persist_in_memory_then_reopen(self, tmp_path: Path) -> None:
        """In-memory store can be persisted and then reopened."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        store.upsert_nodes([EntityNode(name=f"e_{i}", label="entity", properties={"idx": i}) for i in range(20)])
        store.upsert_relations(
            [
                Relation(
                    label="REL",
                    source_id=f"e_{i}",
                    target_id=f"e_{i + 1}",
                    properties={},
                )
                for i in range(10)
            ]
        )

        persist_path = str(tmp_path / "saved.db")
        store.persist(persist_path)
        store.close()

        store2 = GrafeoPropertyGraphStore(db_path=persist_path, embedding_dimensions=DIMS)
        assert store2.node_count == 20
        assert store2.edge_count == 10
        store2.close()


class TestFileBackedStore:
    """A file-backed store closed and reopened between updates, deletes and lookups."""

    @LAYOUTS
    def test_updates_and_deletes_survive_reopen(self, tmp_path: Path, filename: str) -> None:
        db_path = str(tmp_path / filename)
        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            _populate(store)
            store.delete(ids=["Bob"])  # has two relations
            store.upsert_nodes([EntityNode(name="Alice", label="person", properties={"age": 31}, embedding=_vec(0))])

        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            assert store._id_cache == {}
            assert store.client.has_property_index("li_id")
            assert {n.id for n in store.get()} == {"Alice", "Acme", "chunk-1"}
            assert store.get(ids=["Alice"])[0].properties["age"] == 31
            assert _triplets(store) == {("Alice", "MENTIONS", "chunk-1")}
            assert _nearest(store, 2) == "Acme"
            assert _nearest(store, 3) == "chunk-1"

            # Writes after a reopen find the stored nodes and relations instead of duplicating them.
            store.upsert_nodes([EntityNode(name="Acme", label="company", properties={"industry": "ai"})])
            store.upsert_relations([Relation(label="MENTIONS", source_id="Alice", target_id="chunk-1")])
            assert (store.node_count, store.edge_count) == (3, 1)
            store.delete(entity_names=["Alice"])

        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            assert {n.id for n in store.get()} == {"Acme", "chunk-1"}
            assert store.get(ids=["Acme"])[0].properties["industry"] == "ai"
            assert store.edge_count == 0
            assert _nearest(store, 3) == "chunk-1"

    @LAYOUTS
    def test_reopen_with_other_dimensions_warns(self, tmp_path: Path, filename: str) -> None:
        """The vector index cannot be rebuilt for 4-dimensional embeddings with dimensions=8:
        this used to be silent, leaving vector_query() empty without explanation."""
        db_path = str(tmp_path / filename)
        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            store.upsert_nodes([EntityNode(name="a", embedding=_vec(0))])

        with pytest.warns(UserWarning, match="embedding_dimensions"):
            store = GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS * 2)
        with store:
            nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=[0.1] * DIMS * 2, similarity_top_k=1))
            assert nodes == []
            assert [n.id for n in store.get()] == ["a"]

    def test_persist_in_memory_store_to_grafeo_file(self, tmp_path: Path) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        path = str(tmp_path / "snapshot.grafeo")
        store.persist(path)
        store.close()

        with GrafeoPropertyGraphStore(db_path=path, embedding_dimensions=DIMS) as reopened:
            assert reopened.client.has_property_index("li_id")
            assert reopened.node_count == 4
            assert _nearest(reopened, 1) == "Bob"
            assert ("Bob", "WORKS_AT", "Acme") in _triplets(reopened)


class TestDatabaseWrittenByGrafeo0544:
    """A database written on grafeo 0.5.44 keeps working on the installed grafeo
    (on 0.6.0 its first open migrates the file to the new format)."""

    def test_open_query_and_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / FIXTURE_0_5_44.name
        shutil.copyfile(FIXTURE_0_5_44, db_path)

        with GrafeoPropertyGraphStore(db_path=str(db_path), embedding_dimensions=DIMS) as store:
            assert {n.id for n in store.get()} == {"Alice", "Bob", "Acme", "chunk-1"}
            assert _triplets(store) == {
                ("Alice", "KNOWS", "Bob"),
                ("Bob", "WORKS_AT", "Acme"),
                ("Alice", "MENTIONS", "chunk-1"),
            }
            [chunk] = store.get(ids=["chunk-1"])
            assert isinstance(chunk, ChunkNode)
            assert chunk.text == "Alice knows Bob."
            assert store.get_triplets(relation_names=["KNOWS"])[0][1].properties == {"since": 2020}
            assert _nearest(store, 1) == "Bob"
            rel_map = store.get_rel_map(store.get(ids=["Bob"]), depth=1)
            assert {(t[0].id, t[1].label, t[2].id) for t in rel_map} == {
                ("Alice", "KNOWS", "Bob"),
                ("Bob", "WORKS_AT", "Acme"),
            }

            store.upsert_nodes([EntityNode(name="Bob", label="person", properties={"age": 26}, embedding=_vec(1))])
            store.delete(ids=["chunk-1"])
            store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"v": 2})])
            assert (store.node_count, store.edge_count) == (3, 2)

        with GrafeoPropertyGraphStore(db_path=str(db_path), embedding_dimensions=DIMS) as store:
            assert store.get(ids=["Bob"])[0].properties["age"] == 26
            assert _triplets(store) == {("Alice", "KNOWS", "Bob"), ("Bob", "WORKS_AT", "Acme")}
            assert _nearest(store, 0) == "Alice"


if __name__ == "__main__":
    # Regenerates FIXTURE_0_5_44; run it only with grafeo 0.5.44 installed.
    import grafeo

    if grafeo.__version__ != "0.5.44":
        sys.exit(f"the fixture must be written by grafeo 0.5.44, found {grafeo.__version__}")
    FIXTURE_0_5_44.parent.mkdir(exist_ok=True)
    FIXTURE_0_5_44.unlink(missing_ok=True)
    with GrafeoPropertyGraphStore(db_path=str(FIXTURE_0_5_44), embedding_dimensions=DIMS) as fixture_store:
        _populate(fixture_store)
