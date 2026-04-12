from __future__ import annotations

from grafeo_llamaindex import GrafeoPropertyGraphStore


class TestRepr:
    def test_empty(self, store: GrafeoPropertyGraphStore) -> None:
        r = repr(store)
        assert "nodes=0" in r
        assert "edges=0" in r

    def test_populated(self, populated_store: GrafeoPropertyGraphStore) -> None:
        r = repr(populated_store)
        assert "nodes=3" in r
        assert "edges=2" in r

    def test_contains_persistent(self, store: GrafeoPropertyGraphStore) -> None:
        r = repr(store)
        assert "persistent=False" in r


class TestSummary:
    def test_empty(self, store: GrafeoPropertyGraphStore) -> None:
        s = store.summary()
        assert "Nodes: 0" in s
        assert "Edges: 0" in s

    def test_populated(self, populated_store: GrafeoPropertyGraphStore) -> None:
        s = populated_store.summary()
        assert "Nodes: 3" in s
        assert "Edges: 2" in s
        assert "person" in s
        assert "company" in s
        assert "KNOWS" in s
        assert "WORKS_AT" in s

    def test_dedup_enabled(self) -> None:
        store = GrafeoPropertyGraphStore(dedup_threshold=0.9)
        s = store.summary()
        assert "Dedup: enabled (threshold=0.9)" in s

    def test_dedup_disabled(self, store: GrafeoPropertyGraphStore) -> None:
        s = store.summary()
        assert "Dedup: disabled" in s

    def test_persistent_status(self, store: GrafeoPropertyGraphStore) -> None:
        s = store.summary()
        assert "Persistent: False" in s


class TestNeighbors:
    def test_by_name(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.neighbors("Alice")
        assert len(triplets) >= 1
        labels = {t[1].label for t in triplets}
        assert "KNOWS" in labels

    def test_by_id(self, populated_store: GrafeoPropertyGraphStore) -> None:
        # Get Alice's actual ID (which differs from name)
        alice_nodes = populated_store.get(properties={"name": "Alice"})
        assert len(alice_nodes) == 1
        triplets = populated_store.neighbors(alice_nodes[0].id)
        assert len(triplets) >= 1

    def test_nonexistent(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.neighbors("Nobody")
        assert triplets == []

    def test_depth(self, populated_store: GrafeoPropertyGraphStore) -> None:
        # depth=2 from Alice should reach Acme via Bob
        triplets = populated_store.neighbors("Alice", depth=2)
        targets = {t[2].id for t in triplets}
        # Bob is direct, Acme is 2-hop
        bob_nodes = populated_store.get(properties={"name": "Bob"})
        acme_nodes = populated_store.get(properties={"name": "Acme"})
        assert bob_nodes[0].id in targets
        assert acme_nodes[0].id in targets

    def test_limit(self, populated_store: GrafeoPropertyGraphStore) -> None:
        triplets = populated_store.neighbors("Alice", depth=2, limit=1)
        assert len(triplets) <= 1


class TestCounts:
    def test_empty(self, store: GrafeoPropertyGraphStore) -> None:
        assert store.node_count == 0
        assert store.edge_count == 0

    def test_populated(self, populated_store: GrafeoPropertyGraphStore) -> None:
        assert populated_store.node_count == 3
        assert populated_store.edge_count == 2
