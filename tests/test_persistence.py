from __future__ import annotations

from pathlib import Path

import grafeo
import pytest
from llama_index.core import StorageContext
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4

# Since grafeo 0.6 every database is a single file, whatever its extension. grafeo 0.5 made a WAL
# directory for a path without the .grafeo extension; databases written by 0.5 are in test_migration.py.
LAYOUTS = pytest.mark.parametrize("filename", ["kg.grafeo", "kg.db"], ids=["grafeo_extension", "other_extension"])


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
        db_path = str(tmp_path / "persist.grafeo")

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
        db_path = str(tmp_path / "persist_vec.grafeo")

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
        db_path = str(tmp_path / "rediscover.grafeo")
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

        persist_path = str(tmp_path / "saved.grafeo")
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
            assert store.get(ids=["chunk-1"])[0].embedding == pytest.approx(_vec(3))

            # Writes after a reopen find the stored nodes and relations instead of duplicating them.
            store.upsert_nodes([EntityNode(name="Acme", label="company", properties={"industry": "ai"})])
            store.upsert_relations([Relation(label="MENTIONS", source_id="Alice", target_id="chunk-1")])
            assert (store.node_count, store.edge_count) == (3, 1)
            store.delete(entity_names=["Alice"])

        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            assert {n.id for n in store.get()} == {"Acme", "chunk-1"}
            assert store.get(ids=["Acme"])[0].properties["industry"] == "ai"
            assert store.edge_count == 0
            assert store.get(ids=["chunk-1"])[0].embedding == pytest.approx(_vec(3))

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


class TestSingleFile:
    """grafeo 0.6 stores a database as one file, with its write-ahead log beside it while open."""

    @LAYOUTS
    def test_database_is_one_file(self, tmp_path: Path, filename: str) -> None:
        db_path = tmp_path / filename
        with GrafeoPropertyGraphStore(db_path=str(db_path), embedding_dimensions=DIMS) as store:
            _populate(store)
            assert db_path.is_file()
            assert (tmp_path / f"{filename}.wal").is_dir()
        assert sorted(p.name for p in tmp_path.iterdir()) == [filename]
        assert db_path.is_file()

    @pytest.mark.parametrize("contents", [[], ["notes.txt"]], ids=["empty", "not_a_database"])
    def test_directory_is_refused(self, tmp_path: Path, contents: list[str]) -> None:
        """grafeo 0.5 created a WAL-directory database in an empty directory; 0.6 refuses any
        directory that is not a 0.5 database, and changes nothing in it."""
        for name in contents:
            (tmp_path / name).write_text("x")
        with pytest.raises(grafeo.GrafeoError, match="not a database file"):
            GrafeoPropertyGraphStore(db_path=str(tmp_path), embedding_dimensions=DIMS)
        assert sorted(p.name for p in tmp_path.iterdir()) == contents


class TestPersist:
    def test_persist_again_replaces_the_file(self, tmp_path: Path) -> None:
        """grafeo 0.6's save() refuses an existing path; persist() replaces the earlier copy."""
        path = tmp_path / "snapshot.grafeo"
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        store.persist(str(path))
        store.delete(ids=["Bob"])
        store.persist(str(path))
        store.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["snapshot.grafeo"]

        with GrafeoPropertyGraphStore(db_path=str(path), embedding_dimensions=DIMS) as reopened:
            assert {n.id for n in reopened.get()} == {"Alice", "Acme", "chunk-1"}
            assert _triplets(reopened) == {("Alice", "MENTIONS", "chunk-1")}

    def test_persist_to_a_directory_raises(self, tmp_path: Path) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        with pytest.raises(IsADirectoryError):
            store.persist(str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_persist_of_a_file_backed_store_does_nothing(self, tmp_path: Path) -> None:
        with GrafeoPropertyGraphStore(db_path=str(tmp_path / "kg.grafeo"), embedding_dimensions=DIMS) as store:
            store.persist(str(tmp_path / "other.grafeo"))
        assert not (tmp_path / "other.grafeo").exists()

    def test_storage_context_persists_twice(self, tmp_path: Path) -> None:
        """LlamaIndex's StorageContext.persist() calls persist() on the graph store every time."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        context = StorageContext.from_defaults(property_graph_store=store)
        context.persist(persist_dir=str(tmp_path))
        store.upsert_nodes([EntityNode(name="Dora", label="person")])
        context.persist(persist_dir=str(tmp_path))
        store.close()

        # LlamaIndex names the file property_graph_store.json; the database is saved as .grafeo instead
        assert not (tmp_path / "property_graph_store.json").exists()
        saved = tmp_path / "property_graph_store.grafeo"
        with GrafeoPropertyGraphStore(db_path=str(saved), embedding_dimensions=DIMS) as reopened:
            assert reopened.node_count == 5

    def test_storage_context_reloads(self, tmp_path: Path) -> None:
        """StorageContext.from_defaults(persist_dir=...) loads after persist(); it failed on the .json file."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        StorageContext.from_defaults(property_graph_store=store).persist(persist_dir=str(tmp_path))
        store.close()

        StorageContext.from_defaults(persist_dir=str(tmp_path))  # no store passed: must not read the database
        saved = tmp_path / "property_graph_store.grafeo"
        with GrafeoPropertyGraphStore(db_path=str(saved), embedding_dimensions=DIMS) as reopened:
            context = StorageContext.from_defaults(persist_dir=str(tmp_path), property_graph_store=reopened)
            assert context.property_graph_store is reopened
            assert reopened.node_count == 4

    def test_persist_json_path_saves_grafeo(self, tmp_path: Path) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _populate(store)
        store.persist(str(tmp_path / "graph.json"))
        assert not (tmp_path / "graph.json").exists()
        assert (tmp_path / "graph.grafeo").is_file()

    def test_db_path_accepts_pathlib_path(self, tmp_path: Path) -> None:
        with GrafeoPropertyGraphStore(db_path=tmp_path / "kg.grafeo", embedding_dimensions=DIMS) as store:
            _populate(store)
        with GrafeoPropertyGraphStore(db_path=tmp_path / "kg.grafeo", embedding_dimensions=DIMS) as store:
            assert store.node_count == 4
