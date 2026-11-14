"""Engine contract: the grafeo behaviors grafeo-llamaindex relies on.

Each test pins one behavior of the ``grafeo`` package through the exact calls, query shapes and
result formats that ``GrafeoPropertyGraphStore`` and ``GrafeoPGRetriever`` use. When a grafeo
release changes one of them, the failing test names the behavior and the code that depends on it,
instead of a store test failing further downstream. The scheduled "grafeo latest" workflow runs
this module against the newest grafeo on PyPI, pre-releases included.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import grafeo
import pytest

DIMS = 4


def _broke(used_by: str, expected: str) -> str:
    return (
        f"grafeo {grafeo.__version__} changed a behavior grafeo-llamaindex relies on ({used_by}): expected {expected}"
    )


def _rows(db: grafeo.GrafeoDB, query: str, params: dict | None = None) -> list[dict]:
    return list(db.execute(query, params))


@pytest.fixture()
def db() -> grafeo.GrafeoDB:
    return grafeo.GrafeoDB()


@pytest.fixture()
def graph(db: grafeo.GrafeoDB) -> tuple[grafeo.GrafeoDB, int, int, int]:
    """(a:person)-[:KNOWS]->(b:person), returned as (db, a, b, edge) IDs."""
    a = db.create_node(labels=["person"], properties={"li_id": "a", "name": "A", "age": 30})
    b = db.create_node(labels=["person"], properties={"li_id": "b", "name": "B", "age": 25})
    edge = db.create_edge(source_id=a.id, target_id=b.id, edge_type="KNOWS", properties={"since": 2020})
    return db, a.id, b.id, edge.id


class TestModule:
    def test_grafeo_error_is_runtime_error(self) -> None:
        assert issubclass(grafeo.GrafeoError, RuntimeError), _broke(
            "vector_query, _find_similar_node and _rediscover_vector_indexes catch RuntimeError",
            "grafeo.GrafeoError to subclass RuntimeError",
        )

    def test_constructors(self, tmp_path: Path) -> None:
        mem = grafeo.GrafeoDB()
        assert mem.is_persistent is False, _broke("summary(), persist()", "GrafeoDB() to be in-memory")
        assert (mem.node_count, mem.edge_count) == (0, 0), _broke("node_count, edge_count", "int counts")
        disk = grafeo.GrafeoDB(str(tmp_path / "c.grafeo"))
        assert disk.is_persistent is True, _broke("GrafeoPropertyGraphStore(db_path=...)", "GrafeoDB(path) to persist")
        disk.close()


class TestPropertyIndex:
    def test_create_and_query(self, db: grafeo.GrafeoDB) -> None:
        assert db.has_property_index("li_id") is False
        db.create_property_index("li_id")
        assert db.has_property_index("li_id") is True, _broke(
            "GrafeoPropertyGraphStore.__init__", "has_property_index() to report a created index"
        )

    def test_find_nodes_by_property(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, _b, _e = graph
        db.create_property_index("li_id")
        assert db.find_nodes_by_property("li_id", "a") == [a], _broke(
            "_lookup (node ID resolution)", "find_nodes_by_property() to return a list of int node IDs"
        )
        assert db.find_nodes_by_property("li_id", "missing") == [], _broke("_lookup", "[] for no match")


class TestNodes:
    def test_create_and_read(self, db: grafeo.GrafeoDB) -> None:
        props = {"li_id": "x", "name": "X", "n": 3, "f": 1.5, "flag": True, "tags": ["a", "b"]}
        node = db.create_node(labels=["person"], properties=props)
        assert isinstance(node.id, int), _broke("upsert_nodes", "create_node() to return a node with an int .id")
        stored = db.get_node(node.id)
        assert stored is not None
        assert stored.properties() == props, _broke(
            "_grafeo_node_to_labelled", "get_node().properties() to return the stored properties as a dict"
        )

    def test_float_list_is_a_vector(self, db: grafeo.GrafeoDB) -> None:
        embedding = [0.1, 0.2, 0.3, 0.4]
        node = db.create_node(labels=["e"], properties={"embedding": embedding})
        stored = db.get_node(node.id)
        assert stored is not None
        assert stored.properties()["embedding"] == pytest.approx(embedding, rel=1e-6), _broke(
            "upsert_nodes stores LlamaIndex embeddings", "a list of floats to round-trip as a list of floats"
        )

    def test_missing_node(self, db: grafeo.GrafeoDB) -> None:
        assert db.get_node(12345) is None, _broke("_grafeo_node_to_labelled", "get_node() of a missing ID to be None")
        assert db.get_node_labels(12345) is None, _broke(
            "_lookup (cache validation)", "get_node_labels() of a missing ID to be None"
        )

    def test_node_labels(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, _b, _e = graph
        assert db.get_node_labels(a) == ["person"], _broke("_lookup", "get_node_labels() to list the labels")
        unlabelled = db.create_node(labels=[], properties={})
        assert db.get_node_labels(unlabelled.id) == [], _broke("_lookup", "[] (not None) for a node without labels")

    def test_set_node_property(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, _b, _e = graph
        db.set_node_property(a, "age", 31)
        db.set_node_property(a, "embedding", [1.0, 0.0, 0.0, 0.0])
        stored = db.get_node(a)
        assert stored is not None
        assert stored.properties()["age"] == 31, _broke("upsert_nodes (update)", "set_node_property() to overwrite")

    def test_null_property_is_left_out(self, db: grafeo.GrafeoDB) -> None:
        used_by = "upsert_nodes writes LlamaIndex properties as they are, None included"
        node = db.create_node(labels=["x"], properties={"a": None, "b": 1})
        stored = db.get_node(node.id)
        assert stored is not None
        assert stored.properties() == {"b": 1}, _broke(used_by, "a None value to store no property (grafeo 0.6)")
        db.set_node_property(node.id, "b", None)
        stored = db.get_node(node.id)
        assert stored is not None
        assert stored.properties() == {}, _broke(used_by, "set_node_property(None) to remove the property")

    def test_decimal_property(self, db: grafeo.GrafeoDB) -> None:
        node = db.create_node(labels=["x"], properties={"price": Decimal("2.50")})
        stored = db.get_node(node.id)
        assert stored is not None
        price = stored.properties()["price"]
        assert price == Decimal("2.50"), _broke("upsert_nodes, get", "a Decimal property to keep its value")
        assert isinstance(price, Decimal), _broke("upsert_nodes, get", "a Decimal to read back as a Decimal")
        assert str(price) == "2.50", _broke("upsert_nodes, get", "a Decimal to keep its scale")
        rows = _rows(db, "MATCH (n) WHERE n.`price` = $p0 RETURN id(n) AS node_id", {"p0": 2.5})
        assert rows == [{"node_id": node.id}], _broke("get(properties=...)", "a float parameter to equal the decimal")

    def test_node_ids_not_reused(self, db: grafeo.GrafeoDB) -> None:
        first = db.create_node(labels=["x"], properties={})
        db.delete_node(first.id)
        second = db.create_node(labels=["x"], properties={})
        assert second.id != first.id, _broke(
            "_lookup trusts a cached ID whose node still exists",
            "a deleted node's ID not to be given to a new node",
        )


class TestEdges:
    def test_create_and_read(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, e = graph
        assert isinstance(e, int)
        edge = db.get_edge(e)
        assert edge is not None
        assert (edge.source_id, edge.target_id) == (a, b), _broke(
            "_build_triplet", "get_edge().source_id and .target_id to give the stored direction"
        )
        assert edge.edge_type == "KNOWS", _broke("_build_triplet", "get_edge().edge_type to be the edge type")
        assert edge.properties() == {"since": 2020}, _broke("_build_triplet", "get_edge().properties() as a dict")
        assert db.get_edge(12345) is None, _broke("_build_triplet", "get_edge() of a missing ID to be None")

    def test_set_edge_property(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, _a, _b, e = graph
        db.set_edge_property(e, "since", 2021)
        edge = db.get_edge(e)
        assert edge is not None
        assert edge.properties()["since"] == 2021, _broke("upsert_relations (update)", "set_edge_property()")

    def test_any_edge_type_text(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        odd = "WORKS_FOR/AT_(PART'TIME)"
        edge = db.create_edge(source_id=a, target_id=b, edge_type=odd, properties={})
        rows = _rows(db, "MATCH ()-[r]->() WHERE type(r) = $t RETURN id(r) AS edge_id", {"t": odd})
        assert rows == [{"edge_id": edge.id}], _broke(
            "upsert_relations stores LLM-extracted labels as edge types",
            "create_edge() to accept any edge type text and type(r) = $param to match it",
        )


class TestQueries:
    """The query shapes the store sends, with values always passed as parameters."""

    def test_rows_are_dicts_named_by_alias(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        rows = _rows(db, "MATCH (n) RETURN id(n) AS node_id")
        assert sorted(r["node_id"] for r in rows) == [a, b], _broke(
            "get()", "rows as dicts keyed by alias, id(n) as an int"
        )

    def test_none_params(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, *_ = graph
        assert len(_rows(db, "MATCH (n) RETURN id(n) AS node_id", None)) == 2, _broke(
            "structured_query without param_map", "execute(query, None) to run the query"
        )

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        [("name", "A", 1), ("age", 25, 1), ("age", 25.0, 1), ("group", "g", 1), ("missing", None, 0)],
    )
    def test_property_filter(
        self, graph: tuple[grafeo.GrafeoDB, int, int, int], key: str, value: object, expected: int
    ) -> None:
        db, a, _b, _e = graph
        db.set_node_property(a, "group", "g")
        rows = _rows(db, f"MATCH (n) WHERE n.`{key}` = $p0 RETURN id(n) AS node_id", {"p0": value})
        assert len(rows) == expected, _broke(
            "get(properties=...), delete(properties=...), get_triplets(properties=...)",
            "n.`key` = $param, with a backtick-quoted key (also for keywords such as group)",
        )

    def test_triplet_query(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, _a, _b, e = graph
        rows = _rows(
            db,
            "MATCH (src)-[rel]->(tgt) WHERE (src.name IN $entity_names OR tgt.name IN $entity_names) "
            "AND (src.li_id IN $ids OR tgt.li_id IN $ids) AND type(rel) IN $relation_types "
            "AND (src.`age` = $p0 OR tgt.`age` = $p0) "
            "RETURN id(rel) AS rel_id LIMIT 500",
            {"entity_names": ["B"], "ids": ["a"], "relation_types": ["KNOWS"], "p0": 25},
        )
        assert rows == [{"rel_id": e}], _broke(
            "get_triplets", "IN $list filters on either endpoint, type(rel) IN $list and id() columns"
        )

    def test_limit(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        db.create_edge(source_id=b, target_id=a, edge_type="KNOWS", properties={})
        rows = _rows(db, "MATCH (src)-[rel]->(tgt) RETURN id(rel) AS rel_id LIMIT 1")
        assert len(rows) == 1, _broke("get_triplets(limit=...)", "LIMIT to cap the rows")

    def test_existing_edge_lookup(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, e = graph
        query = (
            "MATCH (a)-[r]->(b) WHERE id(a) = $source AND id(b) = $target AND type(r) = $edge_type "
            "RETURN id(r) AS edge_id LIMIT 1"
        )
        found = _rows(db, query, {"source": a, "target": b, "edge_type": "KNOWS"})
        assert found == [{"edge_id": e}], _broke("upsert_relations (dedup)", "the edge between two node IDs")
        reverse = _rows(db, query, {"source": b, "target": a, "edge_type": "KNOWS"})
        assert reverse == [], _broke("upsert_relations (dedup)", "edges to be directed")
        other = _rows(db, query, {"source": a, "target": b, "edge_type": "LIKES"})
        assert other == [], _broke("upsert_relations (dedup)", "type(r) = $param to filter by type")

    def test_hop_query(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, e = graph
        hop = (
            "MATCH (n)-[rel]-(other) WHERE id(n) IN $frontier "
            "RETURN id(rel) AS rel_id, id(other) AS other_id, type(rel) AS rel_type"
        )
        expected = "the edges on either side of the frontier, with the far end and type(rel) as a string"
        assert _rows(db, hop, {"frontier": [a]}) == [{"rel_id": e, "other_id": b, "rel_type": "KNOWS"}], _broke(
            "get_rel_map", expected
        )
        assert _rows(db, hop, {"frontier": [b]}) == [{"rel_id": e, "other_id": a, "rel_type": "KNOWS"}], _broke(
            "get_rel_map", expected
        )

    def test_hop_query_seeks_by_id(self, db: grafeo.GrafeoDB) -> None:
        [row] = _rows(
            db,
            "EXPLAIN MATCH (n)-[rel]-(other) WHERE id(n) IN $frontier "
            "RETURN id(rel) AS rel_id, id(other) AS other_id, type(rel) AS rel_type",
            {"frontier": [0, 1]},
        )
        assert "seek: id" in row["plan"], _broke(
            "get_rel_map runs one hop query per depth level",
            "the frontier to be looked up by id, not every edge expanded",
        )

    def test_gql_decimal_literal(self, db: grafeo.GrafeoDB) -> None:
        [row] = _rows(db, "RETURN 1.50 AS x")
        expected = "a GQL number with a point to be an exact decimal"
        assert row["x"] == Decimal("1.50"), _broke("structured_query(language='gql')", expected)
        assert isinstance(row["x"], Decimal), _broke("structured_query", expected)
        [row] = list(db.execute_cypher("RETURN 1.5 AS x"))
        assert isinstance(row["x"], float), _broke("structured_query (Cypher, the default)", "a Cypher 1.5 as a float")

    def test_unused_params_are_ignored(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, *_ = graph
        rows = _rows(db, "MATCH (n) WHERE n.name = $name RETURN id(n) AS node_id", {"name": "A", "unused": 1})
        assert len(rows) == 1, _broke("structured_query passes param_map as is", "unused parameters to be ignored")

    def test_gremlin_with_params(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, *_ = graph
        rows = list(db.execute_gremlin("g.V().has('name', $name).count()", {"name": "A"}))
        assert rows == [{"count": 1}], _broke("structured_query('g. ...')", "execute_gremlin(query, params)")
        assert list(db.execute_gremlin("g.V().count()", None)) == [{"count": 2}]


class TestDelete:
    def test_detach_delete_by_ids(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        db.create_property_index("li_id")
        db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": [a, 12345]})
        assert (db.node_count, db.edge_count) == (1, 0), _broke(
            "delete()", "DETACH DELETE by id list to remove the nodes and their edges, skipping missing IDs"
        )
        assert db.get_node(a) is None
        assert db.get_node_labels(a) is None
        assert db.find_nodes_by_property("li_id", "a") == [], _broke(
            "delete()", "a deleted node to leave the property index"
        )
        assert db.get_node(b) is not None

    def test_delete_edges_by_type(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        db.create_edge(source_id=a, target_id=b, edge_type="LIKES", properties={})
        db.execute("MATCH ()-[r]->() WHERE type(r) IN $edge_types DELETE r", {"edge_types": ["KNOWS", "NONE"]})
        types = [r["t"] for r in _rows(db, "MATCH ()-[r]->() RETURN type(r) AS t")]
        assert types == ["LIKES"], _broke("delete(relation_names=...)", "only the listed edge types to be deleted")
        assert db.node_count == 2


class TestSchema:
    def test_schema_shape(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, *_ = graph
        schema = db.schema()
        # grafeo 0.6.2 adds a 'properties' list to each label entry; match on name and count.
        labels = {entry["name"]: entry["count"] for entry in schema["labels"]}
        assert labels == {"person": 2}, _broke(
            "get_schema_str, summary, _rediscover_vector_indexes", "schema()['labels'] as [{'name', 'count'}]"
        )
        edge_types = {entry["name"]: entry["count"] for entry in schema["edge_types"]}
        assert edge_types == {"KNOWS": 1}, _broke(
            "get_schema_str, summary", "schema()['edge_types'] as [{'name', 'count'}]"
        )
        assert {"li_id", "name", "age", "since"} <= set(schema["property_keys"]), _broke(
            "get_schema_str, _rediscover_vector_indexes", "schema()['property_keys'] as a list of names"
        )


class TestVectorIndex:
    def _search(self, db: grafeo.GrafeoDB, query: list[float], k: int = 3) -> list[tuple[int, float]]:
        return db.vector_search(label="e", property="embedding", query=query, k=k)

    def test_empty_index_then_auto_indexing(self, db: grafeo.GrafeoDB) -> None:
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        created = db.create_node(labels=["e"], properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
        later = db.create_node(labels=["e"], properties={})
        db.set_node_property(later.id, "embedding", [0.0, 1.0, 0.0, 0.0])
        found = {node_id for node_id, _ in self._search(db, [1.0, 0.0, 0.0, 0.0])}
        assert found == {created.id, later.id}, _broke(
            "upsert_nodes creates a label's vector index once and relies on later writes being indexed",
            "create_node() and set_node_property() to update an existing vector index",
        )

    def test_recreate_existing_index(self, db: grafeo.GrafeoDB) -> None:
        db.create_node(labels=["e"], properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        assert len(self._search(db, [1.0, 0.0, 0.0, 0.0])) == 1, _broke(
            "_ensure_vector_index, _rediscover_vector_indexes", "create_vector_index() on an indexed label to succeed"
        )

    def test_dimension_mismatch_raises(self, db: grafeo.GrafeoDB) -> None:
        db.create_node(labels=["e"], properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
        with pytest.raises(RuntimeError):
            db.create_vector_index(label="e", property="embedding", dimensions=DIMS * 2, metric="cosine")

    def test_wrong_query_size_raises(self, db: grafeo.GrafeoDB) -> None:
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        with pytest.raises(grafeo.GrafeoError):  # 0.5.44 panicked (PanicException is no RuntimeError)
            self._search(db, [1.0, 0.0])

    def test_search_without_index_raises(self, db: grafeo.GrafeoDB) -> None:
        with pytest.raises(RuntimeError):
            self._search(db, [1.0, 0.0, 0.0, 0.0])

    def test_results(self, db: grafeo.GrafeoDB) -> None:
        ids = [
            db.create_node(labels=["e"], properties={"embedding": v}).id
            for v in ([1.0, 0.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        ]
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        results = self._search(db, [1.0, 0.0, 0.0, 0.0], k=2)
        assert [node_id for node_id, _ in results] == ids[:2], _broke(
            "vector_query", "vector_search() to return (node_id, distance) pairs, nearest first, at most k"
        )
        assert all(isinstance(d, float) for _, d in results)

    @pytest.mark.parametrize(
        ("metric", "stored", "query", "expected"),
        [
            ("cosine", [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], 1.0 - 1.0 / math.sqrt(2.0)),  # 1 - cos sim
            ("euclidean", [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], 2.0),
            ("dot_product", [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], -40.0),  # negated dot product
            ("manhattan", [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], 4.0),
        ],
    )
    def test_distance_semantics(
        self, db: grafeo.GrafeoDB, metric: str, stored: list[float], query: list[float], expected: float
    ) -> None:
        db.create_node(labels=["e"], properties={"embedding": stored})
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric=metric)
        [(_, distance)] = self._search(db, query, k=1)
        assert distance == pytest.approx(expected, rel=1e-4, abs=1e-4), _broke(
            "_distance_to_score converts distances to LlamaIndex scores", f"{metric} distance {expected}"
        )

    def test_deleted_node_leaves_index(self, db: grafeo.GrafeoDB) -> None:
        keep = db.create_node(labels=["e"], properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
        drop = db.create_node(labels=["e"], properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": [drop.id]})
        assert [node_id for node_id, _ in self._search(db, [1.0, 0.0, 0.0, 0.0])] == [keep.id], _broke(
            "delete() then vector_query", "DETACH DELETE to remove the node from the vector index"
        )


class TestAlgorithms:
    def test_pagerank(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, _e = graph
        scores = db.algorithms.pagerank()
        assert set(scores) == {a, b}, _broke("GrafeoPGRetriever", "pagerank() as a dict keyed by node ID")
        assert all(isinstance(v, float) for v in scores.values())
        assert scores[b] > scores[a], _broke("GrafeoPGRetriever", "the edge target to rank higher")

    def test_epoch_advances_on_writes_only(self, graph: tuple[grafeo.GrafeoDB, int, int, int]) -> None:
        db, a, b, e = graph
        used_by = "_pagerank keys its cache by current_epoch() to notice writes made through client"

        start = db.current_epoch()
        _rows(db, "MATCH (n) RETURN id(n) AS node_id")
        db.get_node(a)
        db.algorithms.pagerank()
        assert db.current_epoch() == start, _broke(used_by, "reads and pagerank() to keep the epoch")

        writes = [
            lambda: db.create_edge(source_id=b, target_id=a, edge_type="BACK", properties={}),
            lambda: db.delete_edge(e),
            lambda: db.create_node(labels=["x"], properties={}),
            lambda: _rows(db, "INSERT (:x {k: 1})"),
            lambda: _rows(db, "MATCH (n:x) DETACH DELETE n"),
        ]
        for write in writes:
            before = db.current_epoch()
            write()
            assert db.current_epoch() > before, _broke(used_by, "every write, direct or by query, to advance it")


class TestPersistence:
    @pytest.mark.parametrize("name", ["graph.grafeo", "graph.db"], ids=["grafeo_extension", "other_extension"])
    def test_close_and_reopen(self, tmp_path: Path, name: str) -> None:
        path = str(tmp_path / name)
        db = grafeo.GrafeoDB(path)
        assert Path(path).is_file() and (tmp_path / f"{name}.wal").is_dir(), _broke(
            "GrafeoPropertyGraphStore(db_path=...)", "a single database file, with its WAL in <path>.wal/ while open"
        )
        a = db.create_node(labels=["e"], properties={"li_id": "a", "embedding": [1.0, 0.0, 0.0, 0.0]})
        b = db.create_node(labels=["e"], properties={"li_id": "b"})
        db.create_edge(source_id=a.id, target_id=b.id, edge_type="T", properties={"w": 1})
        db.create_property_index("li_id")
        db.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        db.close()

        reopened = grafeo.GrafeoDB(path)
        assert (reopened.node_count, reopened.edge_count) == (2, 1), _broke(
            "GrafeoPropertyGraphStore(db_path=...)", "nodes and edges to survive close() and reopen"
        )
        if not reopened.has_property_index("li_id"):  # the store re-creates a missing index on open
            reopened.create_property_index("li_id")
        assert reopened.find_nodes_by_property("li_id", "a") == [a.id]
        # The store rebuilds vector indexes on open; that must work on a reopened database.
        reopened.create_vector_index(label="e", property="embedding", dimensions=DIMS, metric="cosine")
        assert [n for n, _ in reopened.vector_search(label="e", property="embedding", query=[1.0, 0, 0, 0], k=1)] == [
            a.id
        ]
        reopened.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == [name], _broke(
            "GrafeoPropertyGraphStore.close()", "close() to fold the WAL into the file and remove <path>.wal/"
        )

    def test_save_in_memory_database(self, tmp_path: Path) -> None:
        db = grafeo.GrafeoDB()
        db.create_node(labels=["e"], properties={"li_id": "a"})
        path = str(tmp_path / "saved.grafeo")
        db.save(path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["saved.grafeo"], _broke(
            "persist() moves the saved file over the target", "save() to write one file and nothing beside it"
        )
        with pytest.raises(grafeo.GrafeoError):
            db.save(path)  # persist() saves to a new path and replaces the target itself
        reopened = grafeo.GrafeoDB(path)
        assert reopened.node_count == 1, _broke("persist()", "save(path) to write a database GrafeoDB(path) opens")
        reopened.close()

    def test_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(grafeo.GrafeoError):
            grafeo.GrafeoDB(str(tmp_path))
        assert list(tmp_path.iterdir()) == [], _broke(
            "GrafeoPropertyGraphStore(db_path=<directory>)", "an empty directory to be refused and left as it is"
        )
