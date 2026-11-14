"""Write the grafeo 0.5.44 databases that ``tests/test_migration.py`` opens with grafeo 0.6.

The databases are written by this package's store, in both layouts grafeo 0.5.x had: a single ``.grafeo`` file and
the WAL directory it created for any other path. Run it from an empty scratch directory with grafeo 0.5.44 (the store
code under ``src/`` still runs on 0.5.44), then copy the two outputs into ``tests/data``:

    uv run --no-project --with grafeo==0.5.44 --with llama-index-core python <repo>/tests/data/write_fixtures_0_5_44.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import grafeo
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 4
NAMES = ("store-grafeo-0.5.44.grafeo", "store-grafeo-0.5.44.db")


def vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIMS)]


def populate(store: GrafeoPropertyGraphStore) -> None:
    """Alice-[KNOWS]->Bob-[WORKS_AT]->Acme, Alice-[MENTIONS]->chunk-1, all with embeddings."""
    store.upsert_nodes(
        [
            EntityNode(name="Alice", label="person", properties={"age": 30, "score": 2.5}, embedding=vec(0)),
            EntityNode(name="Bob", label="person", properties={"age": 25, "tags": ["a", "b"]}, embedding=vec(1)),
            EntityNode(name="Acme", label="company", properties={"industry": "tech"}, embedding=vec(2)),
            ChunkNode(id_="chunk-1", text="Alice knows Bob.", properties={"ref_doc_id": "doc-1"}, embedding=vec(3)),
        ]
    )
    store.upsert_relations(
        [
            Relation(label="KNOWS", source_id="Alice", target_id="Bob", properties={"since": 2020}),
            Relation(label="works at", source_id="Bob", target_id="Acme"),
            Relation(label="MENTIONS", source_id="Alice", target_id="chunk-1"),
        ]
    )


def main() -> None:
    if grafeo.__version__ != "0.5.44":
        sys.exit(f"the fixtures must be written by grafeo 0.5.44, found {grafeo.__version__}")
    for name in NAMES:
        if Path(name).exists():
            sys.exit(f"{name} exists: run this in an empty directory")
        with GrafeoPropertyGraphStore(db_path=name, embedding_dimensions=DIMS) as store:
            populate(store)
        print(f"wrote {Path(name).resolve()}")


if __name__ == "__main__":
    main()
