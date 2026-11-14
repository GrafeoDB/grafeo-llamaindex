"""Values the store hands back that grafeo 0.6 changed: exact decimals, the order of nulls, null properties."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from llama_index.core.graph_stores.types import EntityNode

from grafeo_llamaindex import GrafeoPropertyGraphStore


def _assert_decimal(value: object, expected: str) -> None:
    """``value`` is the decimal ``expected``: a Decimal with its scale."""
    assert value == Decimal(expected)
    assert isinstance(value, Decimal)
    assert str(value) == expected


@pytest.fixture()
def priced_store() -> GrafeoPropertyGraphStore:
    store = GrafeoPropertyGraphStore()
    store.upsert_nodes(
        [
            EntityNode(name="Tea", label="product", properties={"price": Decimal("2.50"), "stock": 3}),
            EntityNode(name="Cake", label="product", properties={"price": 4.25, "stock": 1}),
            EntityNode(name="Bag", label="product", properties={"stock": 0}),  # no price
        ]
    )
    return store


class TestDecimals:
    def test_decimal_property_round_trip(self, priced_store: GrafeoPropertyGraphStore) -> None:
        [tea] = priced_store.get(ids=["Tea"])
        _assert_decimal(tea.properties["price"], "2.50")
        [cake] = priced_store.get(ids=["Cake"])
        assert cake.properties["price"] == 4.25
        assert isinstance(cake.properties["price"], float), "a Python float stays a float"

    def test_decimal_property_survives_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "kg.grafeo")
        with GrafeoPropertyGraphStore(db_path=path) as store:
            store.upsert_nodes([EntityNode(name="Tea", properties={"price": Decimal("2.50")})])
        with GrafeoPropertyGraphStore(db_path=path) as store:
            _assert_decimal(store.get(ids=["Tea"])[0].properties["price"], "2.50")

    @pytest.mark.parametrize("value", [2.5, Decimal("2.5"), Decimal("2.500")], ids=["float", "decimal", "scale"])
    def test_filters_match_a_decimal_by_value(self, priced_store: GrafeoPropertyGraphStore, value: object) -> None:
        assert [n.id for n in priced_store.get(properties={"price": value})] == ["Tea"]

    def test_gql_number_with_a_point_is_a_decimal(self, priced_store: GrafeoPropertyGraphStore) -> None:
        [row] = priced_store.structured_query("RETURN 1.50 AS x, 0.1 + 0.2 AS y", language="gql")
        assert row == {"x": Decimal("1.50"), "y": Decimal("0.3")}
        assert str(row["x"]) == "1.50"

    def test_cypher_number_with_a_point_stays_a_float(self, priced_store: GrafeoPropertyGraphStore) -> None:
        """The default language, so text-to-Cypher retrievers keep getting floats from literals."""
        [row] = priced_store.structured_query("RETURN 1.5 AS x, 0.1 + 0.2 AS y")
        assert row == {"x": 1.5, "y": 0.1 + 0.2}
        assert isinstance(row["x"], float)

    def test_decimal_written_by_gql_reads_back(self, priced_store: GrafeoPropertyGraphStore) -> None:
        priced_store.structured_query("MATCH (n) WHERE n.name = 'Cake' SET n.discount = 0.75", language="gql")
        _assert_decimal(priced_store.get(ids=["Cake"])[0].properties["discount"], "0.75")

    def test_decimal_in_query_results(self, priced_store: GrafeoPropertyGraphStore) -> None:
        rows = priced_store.structured_query(
            "MATCH (n:product) WHERE n.price IS NOT NULL RETURN n.name AS name, n.price AS price ORDER BY n.name"
        )
        assert [row["name"] for row in rows] == ["Cake", "Tea"]
        assert rows[0]["price"] == 4.25
        _assert_decimal(rows[1]["price"], "2.50")


class TestNullOrder:
    """grafeo 0.6 GQL puts nulls last in both directions; Cypher keeps openCypher's order (nulls are
    the largest value, so first when descending)."""

    QUERY = "MATCH (n:product) RETURN n.name AS name ORDER BY n.price {direction}"

    @pytest.mark.parametrize(
        ("language", "direction", "expected"),
        [
            ("cypher", "ASC", ["Tea", "Cake", "Bag"]),
            ("cypher", "DESC", ["Bag", "Cake", "Tea"]),
            ("gql", "ASC", ["Tea", "Cake", "Bag"]),
            ("gql", "DESC", ["Cake", "Tea", "Bag"]),
        ],
    )
    def test_default_null_order(
        self, priced_store: GrafeoPropertyGraphStore, language: str, direction: str, expected: list[str]
    ) -> None:
        rows = priced_store.structured_query(self.QUERY.format(direction=direction), language=language)
        assert [row["name"] for row in rows] == expected

    def test_gql_nulls_first_restores_the_0_5_order(self, priced_store: GrafeoPropertyGraphStore) -> None:
        rows = priced_store.structured_query(
            "MATCH (n:product) RETURN n.name AS name ORDER BY n.price DESC NULLS FIRST", language="gql"
        )
        assert [row["name"] for row in rows] == ["Bag", "Cake", "Tea"]


class TestNullProperties:
    """A property set to None does not exist in grafeo 0.6 (0.5 could store a null)."""

    def test_none_property_is_left_out(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="X", properties={"a": None, "b": 1})])
        assert store.get(ids=["X"])[0].properties == {"b": 1}

    def test_update_to_none_removes_the_property(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([EntityNode(name="X", properties={"a": 1, "b": 1})])
        store.upsert_nodes([EntityNode(name="X", properties={"a": None, "b": 2})])
        assert store.get(ids=["X"])[0].properties == {"b": 2}
