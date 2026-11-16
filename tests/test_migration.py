"""Databases this store wrote on grafeo 0.5.44, opened with grafeo 0.6.

grafeo 0.6 migrates a 0.5.x database to its single-file format on the first read-write open and keeps
the old one, unchanged, as ``<path>.pre-0.6``. The fixtures in ``tests/data`` were written by
``tests/data/write_fixtures_0_5_44.py`` with grafeo 0.5.44, in both 0.5 layouts: a ``.grafeo`` file
and the WAL directory 0.5 created for any other path. Each test opens a copy.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4
DATA = Path(__file__).parent / "data"
FIXTURES = pytest.mark.parametrize(
    "fixture", ["store-grafeo-0.5.44.grafeo", "store-grafeo-0.5.44.db"], ids=["grafeo_file", "wal_directory"]
)

TRIPLETS = {("Alice", "KNOWS", "Bob"), ("Bob", "WORKS_AT", "Acme"), ("Alice", "MENTIONS", "chunk-1")}


def _vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIMS)]


def _copy(fixture: str, tmp_path: Path) -> Path:
    source, target = DATA / fixture, tmp_path / fixture
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copyfile(source, target)
    return target


def _same_tree(left: Path, right: Path) -> bool:
    if left.is_file():
        return right.is_file() and filecmp.cmp(left, right, shallow=False)
    comparison = filecmp.dircmp(left, right)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(left, right, comparison.common_files, shallow=False)
    return not mismatch and not errors and all(_same_tree(left / d, right / d) for d in comparison.common_dirs)


def _triplets(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


def _nearest(store: GrafeoPropertyGraphStore, i: int) -> str:
    nodes, _ = store.vector_query(VectorStoreQuery(query_embedding=_vec(i), similarity_top_k=1))
    return nodes[0].id


@FIXTURES
def test_fixture_is_a_grafeo_0_5_database(fixture: str) -> None:
    """Guards the fixtures themselves: a .pre-0.6 copy or a migrated file must not replace them."""
    path = DATA / fixture
    if fixture.endswith(".db"):
        assert (path / "wal").is_dir()
    else:
        assert path.read_bytes()[:5] == b"GRAF\x01", "a grafeo 0.5 file header (0.6 writes format 3)"
    assert not (DATA / f"{fixture}.pre-0.6").exists()


@FIXTURES
class TestMigration:
    def test_open_migrates_and_keeps_the_old_database(self, tmp_path: Path, fixture: str) -> None:
        db_path = _copy(fixture, tmp_path)
        with GrafeoPropertyGraphStore(db_path=str(db_path), embedding_dimensions=DIMS) as store:
            assert store.node_count == 4
        backup = tmp_path / f"{fixture}.pre-0.6"
        assert _same_tree(DATA / fixture, backup), "the .pre-0.6 copy must be the 0.5.44 database, unchanged"
        assert db_path.is_file(), "the migrated database is a single file at the same path"
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted([fixture, backup.name])

    def test_nodes_and_relations(self, tmp_path: Path, fixture: str) -> None:
        with GrafeoPropertyGraphStore(db_path=str(_copy(fixture, tmp_path)), embedding_dimensions=DIMS) as store:
            nodes = {n.id: n for n in store.get()}
            assert set(nodes) == {"Alice", "Bob", "Acme", "chunk-1"}
            assert isinstance(nodes["Alice"], EntityNode)
            assert nodes["Alice"].label == "person"
            assert nodes["Alice"].properties == {"age": 30, "score": 2.5}
            assert nodes["Bob"].properties == {"age": 25, "tags": ["a", "b"]}
            assert nodes["Acme"].properties == {"industry": "tech"}
            chunk = nodes["chunk-1"]
            assert isinstance(chunk, ChunkNode)
            assert chunk.text == "Alice knows Bob."
            assert chunk.properties == {"ref_doc_id": "doc-1"}

            assert _triplets(store) == TRIPLETS
            [knows] = store.get_triplets(relation_names=["KNOWS"])
            assert knows[1].properties == {"since": 2020}
            assert store.get(properties={"age": 25})[0].id == "Bob"

    def test_embeddings_and_vector_search(self, tmp_path: Path, fixture: str) -> None:
        with GrafeoPropertyGraphStore(db_path=str(_copy(fixture, tmp_path)), embedding_dimensions=DIMS) as store:
            assert store._vector_indexed_labels == {"person", "company", "text_chunk"}
            for i, li_id in enumerate(["Alice", "Bob", "Acme", "chunk-1"]):
                [node] = store.get(ids=[li_id])
                assert node.embedding == pytest.approx(_vec(i))
                if li_id != "chunk-1":  # vector_query searches entities only
                    assert _nearest(store, i) == li_id

    def test_rel_map(self, tmp_path: Path, fixture: str) -> None:
        with GrafeoPropertyGraphStore(db_path=str(_copy(fixture, tmp_path)), embedding_dimensions=DIMS) as store:
            rel_map = store.get_rel_map(store.get(ids=["Bob"]), depth=1)
            assert {(t[0].id, t[1].label, t[2].id) for t in rel_map} == {
                ("Alice", "KNOWS", "Bob"),
                ("Bob", "WORKS_AT", "Acme"),
            }
            assert {(t[0].id, t[1].label, t[2].id) for t in store.get_rel_map(store.get(ids=["Acme"]), depth=3)} == (
                TRIPLETS
            )

    def test_writes_after_migration_survive_reopen(self, tmp_path: Path, fixture: str) -> None:
        db_path = str(_copy(fixture, tmp_path))
        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            store.upsert_nodes([EntityNode(name="Bob", label="person", properties={"age": 26}, embedding=_vec(1))])
            store.delete(ids=["chunk-1"])
            store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"v": 2})])
            store.upsert_nodes([EntityNode(name="Dora", label="person", embedding=_vec(3))])
            assert (store.node_count, store.edge_count) == (4, 2)

        with GrafeoPropertyGraphStore(db_path=db_path, embedding_dimensions=DIMS) as store:
            assert store.get(ids=["Bob"])[0].properties["age"] == 26
            assert _triplets(store) == {("Alice", "KNOWS", "Bob"), ("Bob", "WORKS_AT", "Acme")}
            assert store.get_triplets(relation_names=["KNOWS"])[0][1].properties == {"since": 2020, "v": 2}
            assert _nearest(store, 0) == "Alice"
            assert _nearest(store, 3) == "Dora"
        assert (tmp_path / f"{Path(db_path).name}.pre-0.6").exists(), "a later open keeps the .pre-0.6 copy"
