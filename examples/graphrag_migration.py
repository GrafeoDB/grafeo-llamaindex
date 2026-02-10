"""Migrate Microsoft GraphRAG parquet output into Grafeo via LlamaIndex.

Reads the parquet files produced by `graphrag index` and loads entities,
relationships, and communities into a GrafeoPropertyGraphStore.

Usage:
    pip install grafeo-llamaindex pandas pyarrow
    python graphrag_migration.py --input ./graphrag-output --output ./migrated.db

Requires:
    - pandas (for parquet reading)
    - A completed GraphRAG indexing run (produces parquet files in output/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core.graph_stores.types import EntityNode, Relation

from grafeo_llamaindex import GrafeoPropertyGraphStore


def load_entities(graph_store: GrafeoPropertyGraphStore, output_dir: Path) -> int:
    """Load entities from GraphRAG parquet into the graph store."""
    import pandas as pd

    entities_path = output_dir / "create_final_entities.parquet"
    if not entities_path.exists():
        print(f"  Skipping entities: {entities_path} not found")
        return 0

    df = pd.read_parquet(entities_path)
    nodes = []
    for _, row in df.iterrows():
        props: dict = {}
        if "description" in df.columns and pd.notna(row.get("description")):
            props["description"] = str(row["description"])
        if "frequency" in df.columns and pd.notna(row.get("frequency")):
            props["frequency"] = int(row["frequency"])
        if "degree" in df.columns and pd.notna(row.get("degree")):
            props["degree"] = int(row["degree"])

        entity_type = str(row.get("type", "entity")).lower()
        nodes.append(
            EntityNode(
                name=str(row["title"]),
                label=entity_type,
                properties=props,
            )
        )

    if nodes:
        graph_store.upsert_nodes(nodes)
    return len(nodes)


def load_relationships(graph_store: GrafeoPropertyGraphStore, output_dir: Path) -> int:
    """Load relationships from GraphRAG parquet into the graph store."""
    import pandas as pd

    rels_path = output_dir / "create_final_relationships.parquet"
    if not rels_path.exists():
        print(f"  Skipping relationships: {rels_path} not found")
        return 0

    df = pd.read_parquet(rels_path)
    relations = []
    for _, row in df.iterrows():
        props: dict = {}
        if "description" in df.columns and pd.notna(row.get("description")):
            props["description"] = str(row["description"])
        if "weight" in df.columns and pd.notna(row.get("weight")):
            props["weight"] = float(row["weight"])

        relations.append(
            Relation(
                label="RELATED_TO",
                source_id=str(row["source"]),
                target_id=str(row["target"]),
                properties=props,
            )
        )

    if relations:
        graph_store.upsert_relations(relations)
    return len(relations)


def load_communities(graph_store: GrafeoPropertyGraphStore, output_dir: Path) -> int:
    """Load community structure from GraphRAG parquet into the graph store."""
    import pandas as pd

    communities_path = output_dir / "create_final_communities.parquet"
    if not communities_path.exists():
        print(f"  Skipping communities: {communities_path} not found")
        return 0

    df = pd.read_parquet(communities_path)
    community_nodes = []
    for _, row in df.iterrows():
        props: dict = {"level": int(row.get("level", 0))}
        if "size" in df.columns and pd.notna(row.get("size")):
            props["size"] = int(row["size"])

        community_nodes.append(
            EntityNode(
                name=str(row.get("title", f"community-{row['id']}")),
                label="community",
                properties=props,
            )
        )

    if community_nodes:
        graph_store.upsert_nodes(community_nodes)

    # Link communities to their member entities
    member_rels = []
    for _, row in df.iterrows():
        community_name = str(row.get("title", f"community-{row['id']}"))
        entity_ids = row.get("entity_ids", [])
        if isinstance(entity_ids, str):
            import json

            entity_ids = json.loads(entity_ids)
        if not isinstance(entity_ids, list):
            continue
        for entity_id in entity_ids:
            member_rels.append(
                Relation(
                    label="HAS_MEMBER",
                    source_id=community_name,
                    target_id=str(entity_id),
                    properties={},
                )
            )

    if member_rels:
        graph_store.upsert_relations(member_rels)
    return len(community_nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate GraphRAG parquet output into Grafeo")
    parser.add_argument("--input", required=True, help="Path to GraphRAG output directory")
    parser.add_argument("--output", default="./graphrag_migrated.db", help="Path for Grafeo database")
    args = parser.parse_args()

    output_dir = Path(args.input)
    if not output_dir.is_dir():
        parser.error(f"Input directory does not exist: {output_dir}")

    print(f"Migrating GraphRAG output from {output_dir}")
    graph_store = GrafeoPropertyGraphStore(db_path=args.output)

    n_entities = load_entities(graph_store, output_dir)
    print(f"  Loaded {n_entities} entities")

    n_rels = load_relationships(graph_store, output_dir)
    print(f"  Loaded {n_rels} relationships")

    n_communities = load_communities(graph_store, output_dir)
    print(f"  Loaded {n_communities} communities")

    print(f"\nMigration complete. Database saved to {args.output}")
    print(f"Schema: {graph_store.get_schema_str()}")
    graph_store.close()


if __name__ == "__main__":
    main()
