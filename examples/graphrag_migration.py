"""Migrate Microsoft GraphRAG parquet output into Grafeo via LlamaIndex.

Reads the parquet files produced by `graphrag index` and loads entities, relationships,
communities and community reports into a GrafeoPropertyGraphStore:

- each entity becomes an `EntityNode` named by its title, labelled by its type
- each relationship becomes a `RELATED_TO` relation with its description and weight
- each community becomes a `community` node, with its report's title, summary and rank when
  `community_reports.parquet` exists, linked to its member entities by `HAS_MEMBER`

Reads the output of GraphRAG 1.0 and later (`entities.parquet`, ...) and of earlier versions
(`create_final_entities.parquet`, ...). Embeddings stay in GraphRAG's vector store and are not
migrated; embed the nodes again with a LlamaIndex embedding model if needed.

Usage:
    uv add grafeo-llamaindex pandas pyarrow
    uv run python graphrag_migration.py --input ./ragtest/output --output ./migrated.grafeo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core.graph_stores.types import EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore


def _parquet(output_dir: Path, name: str) -> pd.DataFrame | None:
    """Read ``<name>.parquet`` (GraphRAG 1.0+) or ``create_final_<name>.parquet`` (earlier)."""
    for path in (output_dir / f"{name}.parquet", output_dir / f"create_final_{name}.parquet"):
        if path.exists():
            return pd.read_parquet(path)
    print(f"  Skipping {name}: no {name}.parquet in {output_dir}")
    return None


def _present(value: Any) -> bool:
    """True for a value that is set: not None, NaN or an empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return value != ""


def _as_list(value: Any) -> list:
    """A list column value as a list: parquet gives numpy arrays, older output JSON strings."""
    if isinstance(value, str):
        return list(json.loads(value))
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return list(value)


def load_entities(graph_store: GrafeoPropertyGraphStore, output_dir: Path) -> dict[str, str]:
    """Load entities as nodes. Returns each GraphRAG entity ID mapped to the entity's title."""
    df = _parquet(output_dir, "entities")
    if df is None:
        return {}

    nodes = []
    titles: dict[str, str] = {}
    for row in df.to_dict("records"):
        props: dict[str, Any] = {}
        if _present(row.get("description")):
            props["description"] = str(row["description"])
        for key in ("frequency", "degree"):
            if _present(row.get(key)):
                props[key] = int(row[key])

        entity_type = row.get("type")
        label = str(entity_type).lower() if _present(entity_type) else "entity"
        title = str(row["title"])
        nodes.append(EntityNode(name=title, label=label, properties=props))
        if _present(row.get("id")):
            titles[str(row["id"])] = title

    graph_store.upsert_nodes(nodes)
    print(f"  Loaded {len(nodes)} entities")
    return titles


def load_relationships(graph_store: GrafeoPropertyGraphStore, output_dir: Path) -> None:
    """Load relationships as RELATED_TO relations between entities, found by title."""
    df = _parquet(output_dir, "relationships")
    if df is None:
        return

    relations = []
    for row in df.to_dict("records"):
        props: dict[str, Any] = {}
        if _present(row.get("description")):
            props["description"] = str(row["description"])
        if _present(row.get("weight")):
            props["weight"] = float(row["weight"])
        relations.append(
            Relation(label="RELATED_TO", source_id=str(row["source"]), target_id=str(row["target"]), properties=props)
        )

    graph_store.upsert_relations(relations)
    print(f"  Loaded {len(relations)} relationships")


def load_communities(graph_store: GrafeoPropertyGraphStore, output_dir: Path, titles: dict[str, str]) -> None:
    """Load communities as nodes, with their reports, linked to their member entities."""
    df = _parquet(output_dir, "communities")
    if df is None:
        return

    reports: dict[int, dict[str, Any]] = {}
    reports_path = output_dir / "community_reports.parquet"
    if not reports_path.exists():
        reports_path = output_dir / "create_final_community_reports.parquet"
    if reports_path.exists():
        for report in pd.read_parquet(reports_path).to_dict("records"):
            reports[int(report["community"])] = report

    nodes = []
    members = []
    for row in df.to_dict("records"):
        community = int(row.get("community", row.get("id", 0)))
        name = f"community-{community}"
        props: dict[str, Any] = {"community": community, "level": int(row.get("level", 0))}
        if _present(row.get("size")):
            props["size"] = int(row["size"])
        report = reports.get(community)
        if report is not None:
            for key in ("title", "summary"):
                if _present(report.get(key)):
                    props[key] = str(report[key])
            if _present(report.get("rank")):
                props["rank"] = float(report["rank"])
        nodes.append(EntityNode(name=name, label="community", properties=props))

        # Members are listed by entity ID; the store finds relation endpoints by title.
        for entity_id in _as_list(row.get("entity_ids")):
            title = titles.get(str(entity_id))
            if title is not None:
                members.append(Relation(label="HAS_MEMBER", source_id=name, target_id=title))

    graph_store.upsert_nodes(nodes)
    graph_store.upsert_relations(members)
    print(f"  Loaded {len(nodes)} communities ({len(reports)} with reports, {len(members)} memberships)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate GraphRAG parquet output into Grafeo")
    parser.add_argument("--input", required=True, help="GraphRAG output directory (holds entities.parquet)")
    parser.add_argument("--output", default="./graphrag_migrated.grafeo", help="Path for the Grafeo database")
    args = parser.parse_args()

    output_dir = Path(args.input)
    if not output_dir.is_dir():
        parser.error(f"Input directory does not exist: {output_dir}")

    print(f"Migrating GraphRAG output from {output_dir}")
    with GrafeoPropertyGraphStore(db_path=args.output) as graph_store:
        titles = load_entities(graph_store, output_dir)
        load_relationships(graph_store, output_dir)
        load_communities(graph_store, output_dir, titles)

        print(f"\nMigration complete. Database saved to {args.output}")
        print(graph_store.summary())


if __name__ == "__main__":
    main()
