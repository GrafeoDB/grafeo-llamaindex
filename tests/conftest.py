from __future__ import annotations

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore


@pytest.fixture()
def store() -> GrafeoPropertyGraphStore:
    """Fresh in-memory store."""
    return GrafeoPropertyGraphStore()


@pytest.fixture()
def populated_store() -> GrafeoPropertyGraphStore:
    """Store pre-populated with Alice-[KNOWS]->Bob-[WORKS_AT]->Acme."""
    s = GrafeoPropertyGraphStore()
    s.upsert_nodes(
        [
            EntityNode(name="Alice", label="person", properties={"age": 30}),
            EntityNode(name="Bob", label="person", properties={"age": 25}),
            EntityNode(name="Acme", label="company", properties={"industry": "tech"}),
        ]
    )
    s.upsert_relations(
        [
            Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"since": 2020}),
            Relation(label="works at", source_id="Bob", target_id="Acme", properties={}),
        ]
    )
    return s
